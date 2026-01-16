import pandas as pd
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from dotenv import load_dotenv

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
    with mlflow.start_run():
        mlflow.log_param("hyperparams", hyperparams)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("experiment_name", experiment_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_folder = f"experiments/{experiment_name}"
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
        mlflow.sklearn.log_model(sk_model=model, name=f"{model_class.__name__}_{experiment_name}")
        # Evaluate on training set
        y_train_pred = model.predict(X_train_scaled)
        train_metrics = evaluate_model(y_train, y_train_pred, "Training")
        mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
        mlflow.log_metric("train_precision", train_metrics["precision"])
        mlflow.log_metric("train_recall", train_metrics["recall"])
        mlflow.log_metric("train_f1", train_metrics["f1"])
        plot_confusion_matrix(y_train, y_train_pred, 
                            f"{exp_folder}/confusion_matrix_train.png", "Training")
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_train.png")
        
        # Evaluate on validation set
        y_valid_pred = model.predict(X_valid_scaled)
        valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")
        plot_confusion_matrix(y_valid, y_valid_pred, 
                            f"{exp_folder}/confusion_matrix_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_valid.png")
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
        
        print(f"\nExperiment results saved to {exp_folder}/experiment_results.json")
        
        return {
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "exp_folder": exp_folder,
            "model": model,
            "scaler": scaler
        }


# ============ Main ============

if __name__ == "__main__":
    
    # Define experiment config
    data_path = "data/training_data_features_imputed.csv"
    experiment_name = f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    hyperparams = {
        "penalty": None,
        "max_iter": 1000,
        "class_weight": "balanced"
    }
    
    # Run experiment
    results = run_experiment(
        data_path=data_path,
        model_class=LogisticRegression,
        hyperparams=hyperparams,
        experiment_name=experiment_name
    )
    
    print("\n" + "="*50)
    print("Experiment Complete!")
    print(f"Results saved to: {results['exp_folder']}")
