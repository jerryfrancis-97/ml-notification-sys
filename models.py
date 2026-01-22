import pandas as pd
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
from dotenv import load_dotenv
from analysis_utils import export_confusion_matrix_splits

load_dotenv(".env")

# ============ Reusable Utility Functions ============

def load_data(path):
    """Load and sort data by timestamp"""
    df = pd.read_csv(path)
    df = df.sort_values("timestamp")
    return df


def get_feature_columns():
    """Return list of feature columns for model input"""
    return [
        "hour", "day_of_week", "is_weekend",
        "num_notifications_last_24h", "delay_since_last_open_notification",
        "user_open_rate", "user_hour_open_rate", "hour_sin", "hour_cos"
    ]


def time_based_split(df, train_ratio=0.7, valid_ratio=0.2):
    """Split data sequentially without shuffle (time-based)"""
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    
    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]
    
    return train, valid, test


def evaluate_model(y_true, y_pred, split_name=""):
    """Calculate and print classification metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    print(f"\n{split_name} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path, split_name=""):
    """Plot and save confusion matrix as PNG"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Opened', 'Opened'],
                yticklabels=['Not Opened', 'Opened'])
    plt.title(f'Confusion Matrix - {split_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_learning_curve(model, X, y, save_path, cv=5):
    """Plot learning curve showing train/validation scores vs training size"""
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, valid_mean, 'o-', color='orange', label='Cross-Validation Score')
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, 
                     alpha=0.1, color='orange')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title('Learning Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Learning curve saved to {save_path}")


def plot_pr_curve(y_true, y_proba, save_path, split_name=""):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {split_name}')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"PR curve saved to {save_path}")
    return pr_auc


def plot_roc_curve(y_true, y_proba, save_path, split_name=""):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {split_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"ROC curve saved to {save_path}")
    return roc_auc


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
        
        exp_folder = f"experiments/{experiment_name}/{timestamp}"
        os.makedirs(exp_folder, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Running Experiment: {experiment_name}")
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
        
        # Prepare X, y
        features = get_feature_columns()
        X_train, y_train = train[features], train["opened"]
        X_valid, y_valid = valid[features], valid["opened"]
        X_test, y_test = test[features], test["opened"]
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
            "data_splits": {
                "train": len(train),
                "valid": len(valid),
                "test": len(test)
            },
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
        "penalty": None,
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
    print(f"Results saved to: experiments/{experiment_name}/{timestamp}")
