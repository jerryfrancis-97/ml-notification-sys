import os
from datetime import datetime
import optuna
import lightgbm as lgb
import mlflow
from sklearn.metrics import f1_score

# Import from utility modules
from training.data_utils import DataLoader, evaluate_model, get_dvc_hash
from viz.viz_utils import (
    plot_confusion_matrix, plot_pr_curve, plot_roc_curve,
    plot_training_history, plot_accuracy_curves, plot_f1_curves,
    plot_precision_curves, plot_recall_curves, plot_feature_importance,
    compute_metrics_per_round
)
from viz.analysis_utils import export_confusion_matrix_splits


DATA_PATH = "data/training_data_features_imputed.csv"

# Use DataLoader for data preparation
data_loader = DataLoader(DATA_PATH)
data_loader.load_data()
train, valid, test = data_loader.time_based_split()
features = data_loader.get_features()

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.note.content", "best_lgbm_from_hpo")
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_from_optuna", study.best_value)
        
        # Log DVC data hash for data lineage tracking
        dvc_info = get_dvc_hash(DATA_PATH)
        if dvc_info:
            mlflow.log_param("data_dvc_md5", dvc_info['md5'])
            mlflow.log_param("data_dvc_size", dvc_info['size'])
            print(f"DVC Data Hash: {dvc_info['md5']}")

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
        
        # Export confusion matrix splits for qualitative analysis
        cm_splits = export_confusion_matrix_splits(
            y_valid, y_valid_pred, valid, plot_dir, "valid"
        )
        for path in cm_splits.values():
            mlflow.log_artifact(path, artifact_path="confusion_matrix_analysis")

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
    
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image("optimization_history.png")
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image("param_importances.png")
    mlflow.log_artifact("optimization_history.png")
    mlflow.log_artifact("param_importances.png")

    return study



if __name__ == "__main__":
    study = run_tuning(n_trials=100)
