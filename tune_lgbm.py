import os
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
import pandas as pd
import mlflow
from sklearn.metrics import f1_score
from models import load_data, get_feature_columns, time_based_split
from models_lgbm import (
    plot_training_history, plot_accuracy_curves, plot_f1_curves,
    plot_precision_curves, plot_recall_curves, plot_feature_importance,
    compute_metrics_per_round
)
from models import plot_confusion_matrix, plot_pr_curve, plot_roc_curve, evaluate_model


DATA_PATH = "data/training_data_features_imputed.csv"
df = load_data(DATA_PATH)
train, valid, test = time_based_split(df)
features = get_feature_columns()

X_train, y_train = train[features].values, train["opened"].values
X_valid, y_valid = valid[features].values, valid["opened"].values

def objective(trial):
    """Objective function for Lightgbm tuning"""
    params = {
        "objective": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 2, 256, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0, step=0.1),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50, step=5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0, step=0.1),
        "early_stopping_rounds": 20,
    }

    model = lgb.LGBMClassifier(
        **params,
    )

    model.fit(X_train, y_train, 
        eval_set=[(X_valid, y_valid)], 
        eval_names=["valid_1"], 
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
     )

    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)

    return f1


def run_tuning(n_trials: int = 100):
    """ Run optuna tuning for lightgbm """

    mlflow.set_experiment("lgbm_tuning")

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_hpo",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10,
         n_min_trials=3)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)


    # print(f"Best parameters: {study.best_params}")
    print(f"Best trial: {study.best_trial}")
    print(f"Best value: {study.best_value}")

    with mlflow.start_run(run_name="best_lgbm_from_hpo"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_from_optuna", study.best_value)

        # Initialize evals_result dict for recording training history
        evals_result = {}
        
        model = lgb.LGBMClassifier(**study.best_params, verbosity=-1)
        model.fit(X_train,
                y_train, 
                eval_set=[(X_train, y_train), (X_valid, y_valid)], 
                eval_names=["training", "valid_1"], 
                eval_metric=['logloss', 'binary_error'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20),
                    lgb.record_evaluation(evals_result)
                ],
        )

        # Create output directory for plots
        plot_dir = f"experiments/lgbm_hpo_best_{study.best_trial.number}"
        os.makedirs(plot_dir, exist_ok=True)

        # Plot training history (loss)
        plot_training_history(evals_result, f"{plot_dir}/training_history.png")
        mlflow.log_artifact(f"{plot_dir}/training_history.png")

        # Plot accuracy curves
        plot_accuracy_curves(evals_result, f"{plot_dir}/accuracy_curves.png")
        mlflow.log_artifact(f"{plot_dir}/accuracy_curves.png")

        # Compute F1, precision, recall per round
        round_metrics = compute_metrics_per_round(model, X_train, y_train, X_valid, y_valid)

        # Plot F1 curves
        plot_f1_curves(round_metrics, f"{plot_dir}/f1_curves.png")
        mlflow.log_artifact(f"{plot_dir}/f1_curves.png")

        # Plot precision curves
        plot_precision_curves(round_metrics, f"{plot_dir}/precision_curves.png")
        mlflow.log_artifact(f"{plot_dir}/precision_curves.png")

        # Plot recall curves
        plot_recall_curves(round_metrics, f"{plot_dir}/recall_curves.png")
        mlflow.log_artifact(f"{plot_dir}/recall_curves.png")

        # Plot feature importance
        importance_df = plot_feature_importance(model, features, f"{plot_dir}/feature_importance.png")
        mlflow.log_artifact(f"{plot_dir}/feature_importance.png")
        importance_df.to_csv(f"{plot_dir}/feature_importance.csv", index=False)
        mlflow.log_artifact(f"{plot_dir}/feature_importance.csv")

        # Evaluate on validation set
        y_valid_pred = model.predict(X_valid)
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")
        mlflow.log_metric("valid_accuracy", valid_metrics["accuracy"])
        mlflow.log_metric("valid_precision", valid_metrics["precision"])
        mlflow.log_metric("valid_recall", valid_metrics["recall"])
        mlflow.log_metric("valid_f1", valid_metrics["f1"])

        # Confusion matrix
        plot_confusion_matrix(y_valid, y_valid_pred, f"{plot_dir}/confusion_matrix_valid.png", "Validation")
        mlflow.log_artifact(f"{plot_dir}/confusion_matrix_valid.png")

        # PR and ROC curves
        valid_pr_auc = plot_pr_curve(y_valid, y_valid_proba, f"{plot_dir}/pr_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{plot_dir}/pr_curve_valid.png")
        mlflow.log_metric("valid_pr_auc", valid_pr_auc)

        valid_roc_auc = plot_roc_curve(y_valid, y_valid_proba, f"{plot_dir}/roc_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{plot_dir}/roc_curve_valid.png")
        mlflow.log_metric("valid_roc_auc", valid_roc_auc)

        # Log model
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name="lgbm_model_hpo_best"
        )
        
        print(f"\nBest model plots saved to: {plot_dir}")
    



    return study



if __name__ == "__main__":
    study = run_tuning(n_trials=100)

    #to show the optimization histo, ry, we can use the following
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()