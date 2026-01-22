import os
from datetime import datetime
import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import mlflow

# Import from utility modules
from data_utils import DataLoader, evaluate_model, get_dvc_hash
from viz_utils import (
    plot_confusion_matrix, plot_pr_curve, plot_roc_curve,
    plot_learning_curve, plot_coefficients
)
from analysis_utils import export_confusion_matrix_splits


DATA_PATH = "data/training_data_features_imputed.csv"

# Use DataLoader for data preparation
data_loader = DataLoader(DATA_PATH)
data_loader.load_data()
train, valid, test = data_loader.time_based_split()
features = data_loader.get_features()

X_train, y_train = train[features].values, train["opened"].values
X_valid, y_valid = valid[features].values, valid["opened"].values

# Scale features (required for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)


def objective(trial):
    """Objective function for Logistic Regression tuning"""
    
    # Suggest penalty type
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])
    
    # Build hyperparameters based on penalty
    params = {
        "max_iter": 1000,
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    
    if penalty is None:
        # No regularization
        params["penalty"] = None
        params["solver"] = "lbfgs"
    elif penalty == "l1":
        params["penalty"] = "l1"
        params["solver"] = "saga"
        params["C"] = trial.suggest_float("C", 1e-4, 10.0, log=True)
    elif penalty == "l2":
        params["penalty"] = "l2"
        params["solver"] = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        params["C"] = trial.suggest_float("C", 1e-4, 10.0, log=True)
    else:  # elasticnet
        params["penalty"] = "elasticnet"
        params["solver"] = "saga"
        params["C"] = trial.suggest_float("C", 1e-4, 10.0, log=True)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    # Train model
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_valid_scaled)
    f1 = f1_score(y_valid, y_pred)
    
    return f1


def run_tuning(n_trials: int = 100):
    """Run Optuna tuning for Logistic Regression"""
    
    mlflow.set_experiment("logreg_tuning")
    
    study = optuna.create_study(
        direction="maximize",
        study_name="logreg_hpo",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10, n_min_trials=3)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best trial: {study.best_trial}")
    print(f"Best value: {study.best_value}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.note.content", "best_logreg_from_hpo")
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_from_optuna", study.best_value)
        
        # Log DVC data hash for data lineage tracking
        dvc_info = get_dvc_hash(DATA_PATH)
        if dvc_info:
            mlflow.log_param("data_dvc_md5", dvc_info['md5'])
            mlflow.log_param("data_dvc_size", dvc_info['size'])
            print(f"DVC Data Hash: {dvc_info['md5']}")
        
        # Reconstruct best params for model
        best_params = study.best_params.copy()
        penalty = best_params.pop("penalty")
        
        model_params = {
            "max_iter": 1000,
            "class_weight": best_params.pop("class_weight"),
        }
        
        if penalty is None:
            model_params["penalty"] = None
            model_params["solver"] = "lbfgs"
        elif penalty == "l1":
            model_params["penalty"] = "l1"
            model_params["solver"] = "saga"
            model_params["C"] = best_params.get("C", 1.0)
        elif penalty == "l2":
            model_params["penalty"] = "l2"
            model_params["solver"] = best_params.get("solver", "lbfgs")
            model_params["C"] = best_params.get("C", 1.0)
        else:  # elasticnet
            model_params["penalty"] = "elasticnet"
            model_params["solver"] = "saga"
            model_params["C"] = best_params.get("C", 1.0)
            model_params["l1_ratio"] = best_params.get("l1_ratio", 0.5)
        
        # Train best model
        model = LogisticRegression(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Create output directory for plots
        plot_dir = f"experiments/logreg_hpo_best_{study.best_trial.number}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot learning curve
        print("\nGenerating learning curve...")
        plot_learning_curve(model, X_train_scaled, y_train, 
                           f"{plot_dir}/learning_curve.png", cv=5)
        mlflow.log_artifact(f"{plot_dir}/learning_curve.png")
        
        # Plot coefficients
        if hasattr(model, 'coef_'):
            coef_df = plot_coefficients(model, features, f"{plot_dir}/coefficients.png")
            mlflow.log_artifact(f"{plot_dir}/coefficients.png")
            coef_df.to_csv(f"{plot_dir}/coefficients.csv", index=False)
            mlflow.log_artifact(f"{plot_dir}/coefficients.csv")
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train_scaled)
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        train_metrics = evaluate_model(y_train, y_train_pred, "Training")
        mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
        mlflow.log_metric("train_precision", train_metrics["precision"])
        mlflow.log_metric("train_recall", train_metrics["recall"])
        mlflow.log_metric("train_f1", train_metrics["f1"])
        
        # Confusion matrix for training
        plot_confusion_matrix(y_train, y_train_pred, 
                            f"{plot_dir}/confusion_matrix_train.png", "Training")
        mlflow.log_artifact(f"{plot_dir}/confusion_matrix_train.png")
        
        # PR and ROC curves for training
        train_pr_auc = plot_pr_curve(y_train, y_train_proba, 
                                     f"{plot_dir}/pr_curve_train.png", "Training")
        mlflow.log_artifact(f"{plot_dir}/pr_curve_train.png")
        mlflow.log_metric("train_pr_auc", train_pr_auc)
        
        train_roc_auc = plot_roc_curve(y_train, y_train_proba, 
                                       f"{plot_dir}/roc_curve_train.png", "Training")
        mlflow.log_artifact(f"{plot_dir}/roc_curve_train.png")
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        
        # Evaluate on validation set
        y_valid_pred = model.predict(X_valid_scaled)
        y_valid_proba = model.predict_proba(X_valid_scaled)[:, 1]
        valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")
        mlflow.log_metric("valid_accuracy", valid_metrics["accuracy"])
        mlflow.log_metric("valid_precision", valid_metrics["precision"])
        mlflow.log_metric("valid_recall", valid_metrics["recall"])
        mlflow.log_metric("valid_f1", valid_metrics["f1"])
        
        # Confusion matrix for validation
        plot_confusion_matrix(y_valid, y_valid_pred, 
                            f"{plot_dir}/confusion_matrix_valid.png", "Validation")
        mlflow.log_artifact(f"{plot_dir}/confusion_matrix_valid.png")
        
        # Export confusion matrix splits for qualitative analysis
        cm_splits = export_confusion_matrix_splits(
            y_valid, y_valid_pred, valid, plot_dir, "valid"
        )
        for path in cm_splits.values():
            mlflow.log_artifact(path, artifact_path="confusion_matrix_analysis")
        
        # PR and ROC curves for validation
        valid_pr_auc = plot_pr_curve(y_valid, y_valid_proba, 
                                     f"{plot_dir}/pr_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{plot_dir}/pr_curve_valid.png")
        mlflow.log_metric("valid_pr_auc", valid_pr_auc)
        
        valid_roc_auc = plot_roc_curve(y_valid, y_valid_proba, 
                                       f"{plot_dir}/roc_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{plot_dir}/roc_curve_valid.png")
        mlflow.log_metric("valid_roc_auc", valid_roc_auc)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="logreg_model_hpo_best"
        )
        
        print(f"\nBest model plots saved to: {plot_dir}")
    
    return study


if __name__ == "__main__":
    study = run_tuning(n_trials=100)
    
    # Uncomment to show optimization visualizations
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()
