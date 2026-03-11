import pandas as pd
import os
import json
from datetime import datetime
import lightgbm as lgb
from lightgbm import LGBMClassifier
import mlflow
from dotenv import load_dotenv

# Import from utility modules
from training.data_utils import DataLoader, evaluate_model, get_dvc_hash
from viz.viz_utils import (
    plot_confusion_matrix, plot_pr_curve, plot_roc_curve,
    plot_feature_importance, plot_training_history, plot_accuracy_curves,
    compute_metrics_per_round, plot_f1_curves, plot_precision_curves, plot_recall_curves,
    plot_loss_learning_curve_lgbm
)
from viz.analysis_utils import export_confusion_matrix_splits

load_dotenv(".env")


# ============ Experiment Setup ============

def run_lgbm_experiment(data_path, hyperparams, experiment_name):
    """
    Run a complete LightGBM experiment with given data and hyperparameters.
    Saves all results to an experiment-specific folder and logs to MLflow.
    
    Args:
        data_path: Path to the feature data CSV
        hyperparams: dict of hyperparameters for LightGBM
        experiment_name: Name for the experiment
    
    Returns:
        dict with train/valid metrics and experiment path
    """
    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("hyperparams", hyperparams)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("model_type", "LightGBM")
        
        # Log DVC data hash for data lineage tracking
        dvc_info = get_dvc_hash(data_path)
        if dvc_info:
            mlflow.log_param("data_dvc_md5", dvc_info['md5'])
            mlflow.log_param("data_dvc_size", dvc_info['size'])
            print(f"DVC Data Hash: {dvc_info['md5']}")
        
        # Create experiment folder
        exp_folder = f"experiments/{experiment_name}/{timestamp}"
        os.makedirs(exp_folder, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Running LightGBM Experiment: {experiment_name}")
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
        
        # Prepare X, y (no scaling needed for tree-based models)
        X_train, y_train = data_loader.get_X_y("train")
        X_valid, y_valid = data_loader.get_X_y("valid")
        X_test, y_test = data_loader.get_X_y("test")
        
        # Convert to numpy arrays for LightGBM
        X_train, y_train = X_train.values, y_train.values
        X_valid, y_valid = X_valid.values, y_valid.values
        X_test, y_test = X_test.values, y_test.values
        
        mlflow.log_param("features", features)
        
        # Extract early stopping rounds from hyperparams
        hyperparams_copy = hyperparams.copy()
        early_stopping_rounds = hyperparams_copy.pop("early_stopping_rounds", None)
        
        # Train LightGBM model
        print("\nTraining LightGBM model...")
        model = LGBMClassifier(**hyperparams_copy)
        
        # Fit with early stopping using validation set
        evals_result = {}
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_names=['training', 'valid_1'],
            eval_metric=['logloss', 'binary_error'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds or 20),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        # Log model to MLflow with registration
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name="lgbm_model"
        )
        
        # Plot training history (loss)
        print("\nGenerating training history plots...")
        plot_training_history(evals_result, f"{exp_folder}/training_history.png")
        mlflow.log_artifact(f"{exp_folder}/training_history.png")
        
        # Plot accuracy curves
        plot_accuracy_curves(evals_result, f"{exp_folder}/accuracy_curves.png")
        mlflow.log_artifact(f"{exp_folder}/accuracy_curves.png")
        
        # Plot loss learning curve (loss vs training set size)
        print("Generating loss learning curve...")
        plot_loss_learning_curve_lgbm(
            X_train, y_train, X_valid, y_valid, hyperparams,
            f"{exp_folder}/learning_curve_loss.png"
        )
        mlflow.log_artifact(f"{exp_folder}/learning_curve_loss.png")
        
        # Compute F1, precision, recall per round
        print("Computing F1, precision, recall per round...")
        round_metrics = compute_metrics_per_round(model, X_train, y_train, X_valid, y_valid)
        
        # Plot F1 curves
        plot_f1_curves(round_metrics, f"{exp_folder}/f1_curves.png")
        mlflow.log_artifact(f"{exp_folder}/f1_curves.png")
        
        # Plot precision curves
        plot_precision_curves(round_metrics, f"{exp_folder}/precision_curves.png")
        mlflow.log_artifact(f"{exp_folder}/precision_curves.png")
        
        # Plot recall curves
        plot_recall_curves(round_metrics, f"{exp_folder}/recall_curves.png")
        mlflow.log_artifact(f"{exp_folder}/recall_curves.png")
        
        # Plot feature importance
        print("Generating feature importance plot...")
        importance_df = plot_feature_importance(model, features, 
                                                f"{exp_folder}/feature_importance.png")
        mlflow.log_artifact(f"{exp_folder}/feature_importance.png")
        importance_df.to_csv(f"{exp_folder}/feature_importance.csv", index=False)
        mlflow.log_artifact(f"{exp_folder}/feature_importance.csv")
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_metrics = evaluate_model(y_train, y_train_pred, "Training")
        mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
        mlflow.log_metric("train_precision", train_metrics["precision"])
        mlflow.log_metric("train_recall", train_metrics["recall"])
        mlflow.log_metric("train_f1", train_metrics["f1"])
        
        # Confusion matrix for training
        plot_confusion_matrix(y_train, y_train_pred, 
                            f"{exp_folder}/confusion_matrix_train.png", "Training")
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_train.png")
        
        # PR and ROC curves for training
        train_pr_auc = plot_pr_curve(y_train, y_train_proba, 
                                     f"{exp_folder}/pr_curve_train.png", "Training")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_train.png")
        mlflow.log_metric("train_pr_auc", train_pr_auc)
        
        train_roc_auc = plot_roc_curve(y_train, y_train_proba, 
                                       f"{exp_folder}/roc_curve_train.png", "Training")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_train.png")
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        
        # Evaluate on validation set
        y_valid_pred = model.predict(X_valid)
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")
        mlflow.log_metric("valid_accuracy", valid_metrics["accuracy"])
        mlflow.log_metric("valid_precision", valid_metrics["precision"])
        mlflow.log_metric("valid_recall", valid_metrics["recall"])
        mlflow.log_metric("valid_f1", valid_metrics["f1"])
        
        # Confusion matrix for validation
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
        
        # Log best iteration
        mlflow.log_metric("best_iteration", model.best_iteration_)
        
        # Print feature importance
        print("\nFeature Importance (sorted):")
        print(importance_df.sort_values("importance", ascending=False).to_string(index=False))
        
        # Save experiment config and results
        experiment_results = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "data_path": data_path,
            "model_type": "LightGBM",
            "hyperparams": hyperparams,
            "features": features,
            "data_splits": data_loader.get_split_sizes(),
            "best_iteration": model.best_iteration_,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics
        }
        
        with open(f"{exp_folder}/experiment_results.json", 'w') as f:
            json.dump(experiment_results, f, indent=2)
        mlflow.log_artifact(f"{exp_folder}/experiment_results.json")
        
        print(f"\nExperiment results saved to {exp_folder}/experiment_results.json")
        
        return {
            "exp_folder": exp_folder,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "best_iteration": model.best_iteration_
        }


# ============ Main ============

if __name__ == "__main__":
    
    # Define experiment config
    data_path = "data/training_data_features_imputed.csv"
    experiment_name = "notification_lgbm"
    
    hyperparams = {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "class_weight": "balanced",
        "verbose": -1,
        "early_stopping_rounds": 20
    }
    
    # Run experiment
    results = run_lgbm_experiment(
        data_path=data_path,
        hyperparams=hyperparams,
        experiment_name=experiment_name
    )
    
    print("\n" + "="*50)
    print("LightGBM Experiment Complete!")
    if results:
        print(f"Results saved to: {results['exp_folder']}")
        print(f"Best iteration: {results['best_iteration']}")
