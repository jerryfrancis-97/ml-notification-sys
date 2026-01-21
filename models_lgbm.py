import pandas as pd
import os
import json
from datetime import datetime
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from dotenv import load_dotenv

# Import shared utilities from models.py
from models import (
    load_data, get_feature_columns, time_based_split, evaluate_model,
    plot_confusion_matrix, plot_pr_curve, plot_roc_curve
)

load_dotenv(".env")


# ============ LightGBM-Specific Functions ============

def plot_feature_importance(model, feature_names, save_path):
    """Plot LightGBM feature importance as horizontal bar chart"""
    importance = model.feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"], importance_df["importance"], color='steelblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Feature importance plot saved to {save_path}")
    return importance_df


def plot_training_history(evals_result, save_path):
    """Plot training and validation loss curves over boosting rounds"""
    train_loss = evals_result['training']['binary_logloss']
    valid_loss = evals_result['valid_1']['binary_logloss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(valid_loss, label='Validation Loss', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('Binary Log Loss')
    plt.title('LightGBM Training History - Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training history saved to {save_path}")


def plot_accuracy_curves(evals_result, save_path):
    """Plot training and validation accuracy curves over boosting rounds"""
    # binary_error is 1 - accuracy, so accuracy = 1 - binary_error
    train_error = evals_result['training']['binary_error']
    valid_error = evals_result['valid_1']['binary_error']
    
    train_acc = [1 - e for e in train_error]
    valid_acc = [1 - e for e in valid_error]
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy', color='blue')
    plt.plot(valid_acc, label='Validation Accuracy', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('Accuracy')
    plt.title('LightGBM Training History - Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Accuracy curves saved to {save_path}")


def compute_metrics_per_round(model, X_train, y_train, X_valid, y_valid):
    """
    Compute F1, precision, and recall at each boosting round for train and validation.
    Returns dict with metrics arrays for plotting.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    n_rounds = model.n_estimators_
    
    metrics = {
        'train_f1': [], 'valid_f1': [],
        'train_precision': [], 'valid_precision': [],
        'train_recall': [], 'valid_recall': []
    }
    
    for i in range(1, n_rounds + 1):
        # Predict using first i trees
        y_train_pred = (model.predict_proba(X_train, num_iteration=i)[:, 1] >= 0.5).astype(int)
        y_valid_pred = (model.predict_proba(X_valid, num_iteration=i)[:, 1] >= 0.5).astype(int)
        
        # Compute metrics
        metrics['train_f1'].append(f1_score(y_train, y_train_pred, zero_division=0))
        metrics['valid_f1'].append(f1_score(y_valid, y_valid_pred, zero_division=0))
        metrics['train_precision'].append(precision_score(y_train, y_train_pred, zero_division=0))
        metrics['valid_precision'].append(precision_score(y_valid, y_valid_pred, zero_division=0))
        metrics['train_recall'].append(recall_score(y_train, y_train_pred, zero_division=0))
        metrics['valid_recall'].append(recall_score(y_valid, y_valid_pred, zero_division=0))
    
    return metrics


def plot_f1_curves(metrics, save_path):
    """Plot training and validation F1 score curves over boosting rounds"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_f1'], label='Training F1', color='blue')
    plt.plot(metrics['valid_f1'], label='Validation F1', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('F1 Score')
    plt.title('LightGBM Training History - F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"F1 curves saved to {save_path}")


def plot_precision_curves(metrics, save_path):
    """Plot training and validation precision curves over boosting rounds"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_precision'], label='Training Precision', color='blue')
    plt.plot(metrics['valid_precision'], label='Validation Precision', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('Precision')
    plt.title('LightGBM Training History - Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Precision curves saved to {save_path}")


def plot_recall_curves(metrics, save_path):
    """Plot training and validation recall curves over boosting rounds"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_recall'], label='Training Recall', color='blue')
    plt.plot(metrics['valid_recall'], label='Validation Recall', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('Recall')
    plt.title('LightGBM Training History - Recall')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Recall curves saved to {save_path}")


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
        
        # Create experiment folder
        exp_folder = f"experiments/{experiment_name}/{timestamp}"
        os.makedirs(exp_folder, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Running LightGBM Experiment: {experiment_name}")
        print(f"Experiment folder: {exp_folder}")
        print(f"{'='*50}")
        
        # Load and split data
        df = load_data(data_path)
        train, valid, test = time_based_split(df)
        mlflow.log_param("data_splits", {
            "train": len(train),
            "valid": len(valid),
            "test": len(test)
        })
        
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
        mlflow.log_param("dataset_shape", df.shape)
        mlflow.log_param("target_distribution", {
            "opened_0": int((df["opened"] == 0).sum()),
            "opened_1": int((df["opened"] == 1).sum())
        })
        
        # Prepare X, y (no scaling needed for tree-based models)
        features = get_feature_columns()
        X_train, y_train = train[features].values, train["opened"].values
        X_valid, y_valid = valid[features].values, valid["opened"].values
        X_test, y_test = test[features].values, test["opened"].values
        mlflow.log_param("features", features)
        
        # Extract early stopping rounds from hyperparams
        early_stopping_rounds = hyperparams.pop("early_stopping_rounds", None)
        
        # Train LightGBM model
        print("\nTraining LightGBM model...")
        model = LGBMClassifier(**hyperparams)
        
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
        
        # Restore early_stopping_rounds for logging
        hyperparams["early_stopping_rounds"] = early_stopping_rounds
        
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
            "data_splits": {
                "train": len(train),
                "valid": len(valid),
                "test": len(test)
            },
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
