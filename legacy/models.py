import pandas as pd
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import mlflow
from dotenv import load_dotenv

# Import from utility modules
from training.data_utils import DataLoader, evaluate_model, load_data, get_feature_columns, time_based_split, get_dvc_hash
from viz.viz_utils import (
    plot_confusion_matrix, plot_learning_curve, plot_loss_learning_curve,
    plot_pr_curve, plot_roc_curve
)
from viz.analysis_utils import export_confusion_matrix_splits

load_dotenv(".env")


# ============ Experiment Setup ============

def run_experiment(data_path, model_class, hyperparams, experiment_name):
    """
    Run a complete experiment with given data, model, and hyperparameters.
    Saves all results to an experiment-specific folder.
    
    Args:
        data_path: Path to the feature data CSV
        model_class: sklearn model class (e.g., LogisticRegression)
        hyperparams: dict of hyperparameters for the model
        experiment_name: Name for the experiment
    
    Returns:
        dict with train/valid metrics and experiment path
    """
    # Create experiment folder with timestamp
    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("hyperparams", hyperparams)
        mlflow.log_param("data_path", data_path)
        
        # Log DVC data hash for data lineage tracking
        dvc_info = get_dvc_hash(data_path)
        if dvc_info:
            mlflow.log_param("data_dvc_md5", dvc_info['md5'])
            mlflow.log_param("data_dvc_size", dvc_info['size'])
            print(f"DVC Data Hash: {dvc_info['md5']}")
        
        exp_folder = f"experiments/{experiment_name}/{timestamp}"
        os.makedirs(exp_folder, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Running Experiment: {experiment_name}")
        print(f"Experiment folder: {exp_folder}")
        print(f"{'='*50}")
        
        # Load and split data using DataLoader
        data_loader = DataLoader(data_path)
        data_loader.load_data()
        train, valid, test = data_loader.time_based_split()
        features = data_loader.get_features()
        
        mlflow.log_param("data_splits", data_loader.get_split_sizes())
        
        # Log dataset as artifact
        mlflow.log_artifact(data_path, artifact_path="datasets")
        
        # Save and log train/valid/test splits
        train.to_csv(f"{exp_folder}/train_split.csv", index=False)
        valid.to_csv(f"{exp_folder}/valid_split.csv", index=False)
        test.to_csv(f"{exp_folder}/test_split.csv", index=False)
        mlflow.log_artifact(f"{exp_folder}/train_split.csv", artifact_path="datasets")
        mlflow.log_artifact(f"{exp_folder}/valid_split.csv", artifact_path="datasets")
        mlflow.log_artifact(f"{exp_folder}/test_split.csv", artifact_path="datasets")
        
        # Log dataset metadata
        mlflow.log_param("dataset_shape", data_loader.df.shape)
        mlflow.log_param("target_distribution", data_loader.get_target_distribution())
        
        # Prepare X, y
        X_train, y_train = data_loader.get_X_y("train")
        X_valid, y_valid = data_loader.get_X_y("valid")
        X_test, y_test = data_loader.get_X_y("test")
        mlflow.log_param("features", features)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = model_class(**hyperparams)
        model.fit(X_train_scaled, y_train)
        mlflow.sklearn.log_model(sk_model=model, name=run_name)
        
        # Plot learning curve (uses cross-validation on training data)
        print("\nGenerating learning curve...")
        plot_learning_curve(model, X_train_scaled, y_train, 
                           f"{exp_folder}/learning_curve.png", cv=5)
        mlflow.log_artifact(f"{exp_folder}/learning_curve.png")
        
        # Plot loss learning curve
        print("Generating loss learning curve...")
        plot_loss_learning_curve(model, X_train_scaled, y_train,
                                f"{exp_folder}/learning_curve_loss.png", cv=5)
        mlflow.log_artifact(f"{exp_folder}/learning_curve_loss.png")
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train_scaled)
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        train_metrics = evaluate_model(y_train, y_train_pred, "Training")
        mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
        mlflow.log_metric("train_precision", train_metrics["precision"])
        mlflow.log_metric("train_recall", train_metrics["recall"])
        mlflow.log_metric("train_f1", train_metrics["f1"])
        plot_confusion_matrix(y_train, y_train_pred, 
                            f"{exp_folder}/confusion_matrix_train.png", "Training")
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_train.png")
        
        # PR and ROC curves for training
        train_pr_auc_score = plot_pr_curve(y_train, y_train_proba, 
                                     f"{exp_folder}/pr_curve_train.png", "Training")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_train.png")
        mlflow.log_metric("train_pr_auc", train_pr_auc_score)
        
        train_roc_auc_score = plot_roc_curve(y_train, y_train_proba, 
                                       f"{exp_folder}/roc_curve_train.png", "Training")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_train.png")
        mlflow.log_metric("train_roc_auc", train_roc_auc_score)
        
        # Evaluate on validation set
        y_valid_pred = model.predict(X_valid_scaled)
        y_valid_proba = model.predict_proba(X_valid_scaled)[:, 1]
        valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")
        plot_confusion_matrix(y_valid, y_valid_pred, 
                            f"{exp_folder}/confusion_matrix_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_valid.png")
        
        # Export confusion matrix splits for qualitative analysis
        cm_splits = export_confusion_matrix_splits(
            y_valid, y_valid_pred, valid, exp_folder, "valid"
        )
        for path in cm_splits.values():
            mlflow.log_artifact(path, artifact_path="confusion_matrix_analysis")
        
        # PR and ROC curves for validation
        valid_pr_auc = plot_pr_curve(y_valid, y_valid_proba, 
                                     f"{exp_folder}/pr_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_valid.png")
        mlflow.log_metric("valid_pr_auc", valid_pr_auc)
        
        valid_roc_auc = plot_roc_curve(y_valid, y_valid_proba, 
                                       f"{exp_folder}/roc_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_valid.png")
        mlflow.log_metric("valid_roc_auc", valid_roc_auc)
        
        # Save coefficients (if model has them)
        if hasattr(model, 'coef_'):
            coefficients_df = pd.DataFrame({
                "feature": features,
                "coefficient": model.coef_[0]
            })
            coefficients_df = coefficients_df.sort_values("coefficient", key=abs, ascending=False)
            coefficients_df.to_csv(f"{exp_folder}/coefficients.csv", index=False)
            mlflow.log_artifact(f"{exp_folder}/coefficients.csv")
            print("\nFeature Coefficients (sorted by importance):")
            print(coefficients_df.to_string(index=False))
        
        mlflow.log_metric("valid_accuracy", valid_metrics["accuracy"])
        mlflow.log_metric("valid_precision", valid_metrics["precision"])
        mlflow.log_metric("valid_recall", valid_metrics["recall"])
        mlflow.log_metric("valid_f1", valid_metrics["f1"])
        
        # Save experiment config and results
        experiment_results = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "data_path": data_path,
            "hyperparams": hyperparams,
            "features": features,
            "data_splits": data_loader.get_split_sizes(),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics
        }
        
        with open(f"{exp_folder}/experiment_results.json", 'w') as f:
            json.dump(experiment_results, f, indent=2)
        mlflow.log_artifact(f"{exp_folder}/experiment_results.json")
        
        print(f"\nExperiment results saved to {exp_folder}/experiment_results.json")


# ============ Main ============

if __name__ == "__main__":
    
    # Define experiment config
    data_path = "data/training_data_features_imputed.csv"
    experiment_name = "notification_logreg"

    hyperparams = {
        "penalty": "l1",
        "solver": "saga" if hyperparams["penalty"] == "l1" else "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced"
    }
    
    # Run experiment
    run_experiment(
        data_path=data_path,
        model_class=LogisticRegression,
        hyperparams=hyperparams,
        experiment_name=experiment_name
    )
    
    print("\n" + "="*50)
    print("Experiment Complete!")
    print(f"Results saved to: experiments/{experiment_name}")
