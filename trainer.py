"""
Unified Trainer Pipeline using sklearn Pipeline.

This module provides a complete ML training pipeline that:
1. Loads feature-engineered data from feature-engg CSV
2. Performs time-based split (BEFORE imputation)
3. Uses sklearn Pipeline to ensure imputer fits ONLY on training data
4. Trains LogisticRegression or LightGBM models
5. Logs all metrics, artifacts, and visualizations to MLflow

Usage:
    python trainer.py --model logreg --data_path data/training_data_features.csv
    python trainer.py --model lgbm --data_path data/training_data_features.csv
"""

import argparse
import os
import json
import yaml
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Import project modules
from data_utils import time_based_split, evaluate_model, get_dvc_hash
from viz_utils import (
    plot_confusion_matrix, plot_learning_curve, plot_loss_learning_curve,
    plot_pr_curve, plot_roc_curve, plot_coefficients,
    plot_feature_importance, plot_training_history, plot_accuracy_curves,
    compute_metrics_per_round, plot_f1_curves, plot_precision_curves, 
    plot_recall_curves, plot_loss_learning_curve_lgbm
)
from analysis_utils import export_confusion_matrix_splits

load_dotenv(".env")


# ============ Config Loading ============

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n[Config] Loaded configuration from: {config_path}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Experiment: {config['experiment']['name']}")
    
    return config


# ============ Feature Columns ============
# These should match the columns created by feature_engg.py

FEATURE_COLUMNS = [
    "num_notifications_last_24h", 
    "delay_since_last_open_notification",
    "user_open_rate", 
    "user_hour_open_rate", 
    "hour_sin", 
    "hour_cos",
    # Interaction features
    "hour_x_user_open_rate",
    "hour_x_user_hour_open_rate",
    "hour_x_num_notifications_last_24h",
    "hour_x_delay_since_last_open_notification"
]

# Columns that need imputation (may have NaN values)
IMPUTE_COLUMNS = ["user_open_rate", "user_hour_open_rate"]


# ============ Custom Transformer ============

class FeatureImputerTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible imputer that:
    - fit(): Learns statistics (mean) from training data ONLY
    - transform(): Applies those SAME training statistics to any data
    
    This prevents data leakage by ensuring validation/test sets use
    only training set statistics for imputation.
    """
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer_ = None
        self.statistics_ = None
    
    def fit(self, X, y=None):
        """Learn imputation statistics from training data ONLY"""
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(X)
        self.statistics_ = self.imputer_.statistics_
        
        print(f"\n[FeatureImputerTransformer] Fitted on training data:")
        print(f"  Strategy: {self.strategy}")
        print(f"  Learned statistics shape: {self.statistics_.shape}")
        
        # Find columns with imputed values (non-trivial statistics)
        for i, stat in enumerate(self.statistics_):
            if not np.isnan(stat):
                print(f"  Feature {i}: {stat:.6f}")
        
        return self
    
    def transform(self, X):
        """Apply TRAINING statistics to transform any data"""
        if self.imputer_ is None:
            raise ValueError("Imputer not fitted. Call fit() first.")
        
        X_transformed = self.imputer_.transform(X)
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for sklearn compatibility"""
        return input_features


# ============ Pipeline Builder ============

def build_logreg_pipeline(hyperparams):
    """
    Build sklearn Pipeline for Logistic Regression.
    
    Pipeline steps:
    1. Imputer: Fill missing values using training mean
    2. Scaler: StandardScaler for feature normalization
    3. Classifier: LogisticRegression
    """
    pipeline = Pipeline([
        ('imputer', FeatureImputerTransformer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**hyperparams))
    ])
    return pipeline


def build_lgbm_pipeline(hyperparams):
    """
    Build sklearn Pipeline for LightGBM.
    
    Pipeline steps:
    1. Imputer: Fill missing values using training mean
    2. Classifier: LGBMClassifier (no scaling needed for tree-based models)
    """
    pipeline = Pipeline([
        ('imputer', FeatureImputerTransformer(strategy='mean')),
        ('classifier', LGBMClassifier(**hyperparams))
    ])
    return pipeline


# ============ Training Functions ============

def train_logreg(pipeline, X_train, y_train, X_valid, y_valid, exp_folder, features):
    """
    Train LogisticRegression pipeline and generate all artifacts.
    """
    print("\n" + "="*50)
    print("Training Logistic Regression Pipeline")
    print("="*50)
    
    # Fit pipeline (imputer + scaler + classifier fitted on train only)
    pipeline.fit(X_train, y_train)
    
    # Get the fitted classifier for additional analysis
    classifier = pipeline.named_steps['classifier']
    scaler = pipeline.named_steps['scaler']
    imputer = pipeline.named_steps['imputer']
    
    # Get transformed data for learning curves (need to transform manually)
    X_train_imputed = imputer.transform(X_train)
    X_train_scaled = scaler.transform(X_train_imputed)
    
    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_valid_pred = pipeline.predict(X_valid)
    y_valid_proba = pipeline.predict_proba(X_valid)[:, 1]
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred, "Training")
    valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")
    
    # Log metrics to MLflow
    for prefix, metrics in [("train", train_metrics), ("valid", valid_metrics)]:
        mlflow.log_metric(f"{prefix}_accuracy", metrics["accuracy"])
        mlflow.log_metric(f"{prefix}_precision", metrics["precision"])
        mlflow.log_metric(f"{prefix}_recall", metrics["recall"])
        mlflow.log_metric(f"{prefix}_f1", metrics["f1"])
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, 
                         f"{exp_folder}/confusion_matrix_train.png", "Training")
    plot_confusion_matrix(y_valid, y_valid_pred, 
                         f"{exp_folder}/confusion_matrix_valid.png", "Validation")
    mlflow.log_artifact(f"{exp_folder}/confusion_matrix_train.png")
    mlflow.log_artifact(f"{exp_folder}/confusion_matrix_valid.png")
    
    # Learning curves (using scaled training data)
    print("Generating learning curve...")
    plot_learning_curve(classifier, X_train_scaled, y_train, 
                       f"{exp_folder}/learning_curve.png", cv=5)
    mlflow.log_artifact(f"{exp_folder}/learning_curve.png")
    
    print("Generating loss learning curve...")
    plot_loss_learning_curve(classifier, X_train_scaled, y_train,
                            f"{exp_folder}/learning_curve_loss.png", cv=5)
    mlflow.log_artifact(f"{exp_folder}/learning_curve_loss.png")
    
    # PR curves
    train_pr_auc = plot_pr_curve(y_train, y_train_proba, 
                                 f"{exp_folder}/pr_curve_train.png", "Training")
    valid_pr_auc = plot_pr_curve(y_valid, y_valid_proba, 
                                 f"{exp_folder}/pr_curve_valid.png", "Validation")
    mlflow.log_artifact(f"{exp_folder}/pr_curve_train.png")
    mlflow.log_artifact(f"{exp_folder}/pr_curve_valid.png")
    mlflow.log_metric("train_pr_auc", train_pr_auc)
    mlflow.log_metric("valid_pr_auc", valid_pr_auc)
    
    # ROC curves
    train_roc_auc = plot_roc_curve(y_train, y_train_proba, 
                                   f"{exp_folder}/roc_curve_train.png", "Training")
    valid_roc_auc = plot_roc_curve(y_valid, y_valid_proba, 
                                   f"{exp_folder}/roc_curve_valid.png", "Validation")
    mlflow.log_artifact(f"{exp_folder}/roc_curve_train.png")
    mlflow.log_artifact(f"{exp_folder}/roc_curve_valid.png")
    mlflow.log_metric("train_roc_auc", train_roc_auc)
    mlflow.log_metric("valid_roc_auc", valid_roc_auc)
    
    # Coefficients
    coef_df = plot_coefficients(classifier, features, f"{exp_folder}/coefficients.png")
    coef_df.to_csv(f"{exp_folder}/coefficients.csv", index=False)
    mlflow.log_artifact(f"{exp_folder}/coefficients.png")
    mlflow.log_artifact(f"{exp_folder}/coefficients.csv")
    
    return {
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "y_valid_pred": y_valid_pred,
        "y_valid_proba": y_valid_proba
    }


def train_lgbm(pipeline, X_train, y_train, X_valid, y_valid, exp_folder, features, hyperparams):
    """
    Train LightGBM pipeline and generate all artifacts.
    
    Note: LightGBM requires special handling for early stopping and eval sets.
    We fit the imputer first, then train LightGBM with eval set.
    """
    print("\n" + "="*50)
    print("Training LightGBM Pipeline")
    print("="*50)
    
    # For LightGBM with early stopping, we need to handle it differently
    # First fit the imputer on training data
    imputer = pipeline.named_steps['imputer']
    imputer.fit(X_train)
    
    # Transform both train and valid using TRAINING statistics
    X_train_imputed = imputer.transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)
    
    print(f"\nImputed training data shape: {X_train_imputed.shape}")
    print(f"Imputed validation data shape: {X_valid_imputed.shape}")
    
    # Get hyperparams and handle early stopping
    lgbm_params = hyperparams.copy()
    early_stopping_rounds = lgbm_params.pop("early_stopping_rounds", 50)
    
    # Create and train LightGBM model with early stopping
    classifier = LGBMClassifier(**lgbm_params)
    
    # Store evaluation results
    evals_result = {}
    
    classifier.fit(
        X_train_imputed, y_train,
        eval_set=[(X_train_imputed, y_train), (X_valid_imputed, y_valid)],
        eval_names=['training', 'valid_1'],
        eval_metric=['binary_logloss', 'binary_error'],
        callbacks=[
            lgb_early_stopping_callback(early_stopping_rounds),
            lgb_record_evaluation_callback(evals_result)
        ]
    )
    
    # Update the pipeline's classifier with the trained one
    pipeline.named_steps['classifier'] = classifier
    
    # Predictions
    y_train_pred = classifier.predict(X_train_imputed)
    y_train_proba = classifier.predict_proba(X_train_imputed)[:, 1]
    y_valid_pred = classifier.predict(X_valid_imputed)
    y_valid_proba = classifier.predict_proba(X_valid_imputed)[:, 1]
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred, "Training")
    valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")
    
    # Log metrics to MLflow
    for prefix, metrics in [("train", train_metrics), ("valid", valid_metrics)]:
        mlflow.log_metric(f"{prefix}_accuracy", metrics["accuracy"])
        mlflow.log_metric(f"{prefix}_precision", metrics["precision"])
        mlflow.log_metric(f"{prefix}_recall", metrics["recall"])
        mlflow.log_metric(f"{prefix}_f1", metrics["f1"])
    
    # Log LightGBM specific info
    mlflow.log_metric("best_iteration", classifier.best_iteration_)
    mlflow.log_metric("n_estimators_used", classifier.n_estimators_)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, 
                         f"{exp_folder}/confusion_matrix_train.png", "Training")
    plot_confusion_matrix(y_valid, y_valid_pred, 
                         f"{exp_folder}/confusion_matrix_valid.png", "Validation")
    mlflow.log_artifact(f"{exp_folder}/confusion_matrix_train.png")
    mlflow.log_artifact(f"{exp_folder}/confusion_matrix_valid.png")
    
    # Training history
    plot_training_history(evals_result, f"{exp_folder}/training_history.png")
    mlflow.log_artifact(f"{exp_folder}/training_history.png")
    
    # Accuracy curves
    plot_accuracy_curves(evals_result, f"{exp_folder}/accuracy_curves.png")
    mlflow.log_artifact(f"{exp_folder}/accuracy_curves.png")
    
    # Compute per-round metrics for F1, precision, recall curves
    print("Computing per-round metrics...")
    metrics_per_round = compute_metrics_per_round(
        classifier, X_train_imputed, y_train, X_valid_imputed, y_valid
    )
    
    plot_f1_curves(metrics_per_round, f"{exp_folder}/f1_curves.png")
    plot_precision_curves(metrics_per_round, f"{exp_folder}/precision_curves.png")
    plot_recall_curves(metrics_per_round, f"{exp_folder}/recall_curves.png")
    mlflow.log_artifact(f"{exp_folder}/f1_curves.png")
    mlflow.log_artifact(f"{exp_folder}/precision_curves.png")
    mlflow.log_artifact(f"{exp_folder}/recall_curves.png")
    
    # Loss learning curve
    print("Generating loss learning curve...")
    plot_loss_learning_curve_lgbm(
        X_train_imputed, y_train, X_valid_imputed, y_valid,
        lgbm_params, f"{exp_folder}/learning_curve_loss.png"
    )
    mlflow.log_artifact(f"{exp_folder}/learning_curve_loss.png")
    
    # PR curves
    train_pr_auc = plot_pr_curve(y_train, y_train_proba, 
                                 f"{exp_folder}/pr_curve_train.png", "Training")
    valid_pr_auc = plot_pr_curve(y_valid, y_valid_proba, 
                                 f"{exp_folder}/pr_curve_valid.png", "Validation")
    mlflow.log_artifact(f"{exp_folder}/pr_curve_train.png")
    mlflow.log_artifact(f"{exp_folder}/pr_curve_valid.png")
    mlflow.log_metric("train_pr_auc", train_pr_auc)
    mlflow.log_metric("valid_pr_auc", valid_pr_auc)
    
    # ROC curves
    train_roc_auc = plot_roc_curve(y_train, y_train_proba, 
                                   f"{exp_folder}/roc_curve_train.png", "Training")
    valid_roc_auc = plot_roc_curve(y_valid, y_valid_proba, 
                                   f"{exp_folder}/roc_curve_valid.png", "Validation")
    mlflow.log_artifact(f"{exp_folder}/roc_curve_train.png")
    mlflow.log_artifact(f"{exp_folder}/roc_curve_valid.png")
    mlflow.log_metric("train_roc_auc", train_roc_auc)
    mlflow.log_metric("valid_roc_auc", valid_roc_auc)
    
    # Feature importance
    importance_df = plot_feature_importance(classifier, features, 
                                           f"{exp_folder}/feature_importance.png")
    importance_df.to_csv(f"{exp_folder}/feature_importance.csv", index=False)
    mlflow.log_artifact(f"{exp_folder}/feature_importance.png")
    mlflow.log_artifact(f"{exp_folder}/feature_importance.csv")
    
    return {
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "y_valid_pred": y_valid_pred,
        "y_valid_proba": y_valid_proba
    }


def lgb_early_stopping_callback(stopping_rounds):
    """Create LightGBM early stopping callback"""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=stopping_rounds, verbose=True)


def lgb_record_evaluation_callback(evals_result):
    """Create LightGBM callback to record evaluation results"""
    from lightgbm import record_evaluation
    return record_evaluation(evals_result)


# ============ Main Training Orchestration ============

def run_training(args):
    """
    Main training function that orchestrates the entire pipeline.
    
    Steps:
    1. Load feature-engineered data
    2. Time-based split (BEFORE imputation)
    3. Build sklearn Pipeline
    4. Train model (imputer fitted on train only)
    5. Generate all metrics and artifacts
    6. Log to MLflow
    """
    
    # Set up experiment
    mlflow.set_experiment(args.experiment)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_run_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        print("\n" + "="*60)
        print(f"Starting Training Pipeline")
        print(f"Model: {args.model.upper()}")
        print(f"Experiment: {args.experiment}")
        print(f"Run: {run_name}")
        print("="*60)
        
        # Log parameters
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("train_ratio", args.train_ratio)
        mlflow.log_param("valid_ratio", args.valid_ratio)
        
        # Log DVC hash for data lineage
        dvc_info = get_dvc_hash(args.data_path)
        if dvc_info:
            mlflow.log_param("data_dvc_md5", dvc_info['md5'])
            mlflow.log_param("data_dvc_size", dvc_info['size'])
            print(f"\nDVC Data Hash: {dvc_info['md5']}")
            print(f"DVC Data Size: {dvc_info['size']}")
        
        # Log config file if used
        if hasattr(args, 'config_path') and args.config_path:
            mlflow.log_artifact(args.config_path, artifact_path="config")
            mlflow.log_param("config_file", args.config_path)
            print(f"Config file: {args.config_path}")
        
        # Create experiment folder
        exp_folder = f"experiments/{args.experiment}/{timestamp}"
        os.makedirs(exp_folder, exist_ok=True)
        mlflow.log_param("experiment_folder", exp_folder)
        
        # ========== Step 1: Load Feature-Engineered Data ==========
        print("\n[Step 1] Loading feature-engineered data...")
        df = pd.read_csv(args.data_path)
        print(f"  Loaded {len(df)} rows from {args.data_path}")
        print(f"  Columns: {list(df.columns)}")
        
        # Log dataset as artifact
        mlflow.log_artifact(args.data_path, artifact_path="datasets")
        
        # Sort by timestamp for proper time-based splitting (if timestamp column exists)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Check for missing values in features
        missing = df[FEATURE_COLUMNS].isnull().sum()
        print(f"\n  Missing values in features (will be imputed):")
        for col, count in missing.items():
            if count > 0:
                print(f"    {col}: {count} ({100*count/len(df):.2f}%)")
        
        # ========== Step 2: Time-Based Split (BEFORE imputation) ==========
        print("\n[Step 2] Performing time-based split (BEFORE imputation)...")
        train, valid, test = time_based_split(df, args.train_ratio, args.valid_ratio)
        
        print(f"  Train: {len(train)} rows ({100*len(train)/len(df):.1f}%)")
        print(f"  Valid: {len(valid)} rows ({100*len(valid)/len(df):.1f}%)")
        print(f"  Test:  {len(test)} rows ({100*len(test)/len(df):.1f}%)")
        
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("valid_size", len(valid))
        mlflow.log_param("test_size", len(test))
        
        # Save splits as artifacts
        train.to_csv(f"{exp_folder}/train_split.csv", index=False)
        valid.to_csv(f"{exp_folder}/valid_split.csv", index=False)
        test.to_csv(f"{exp_folder}/test_split.csv", index=False)
        mlflow.log_artifact(f"{exp_folder}/train_split.csv", artifact_path="datasets/splits")
        mlflow.log_artifact(f"{exp_folder}/valid_split.csv", artifact_path="datasets/splits")
        mlflow.log_artifact(f"{exp_folder}/test_split.csv", artifact_path="datasets/splits")
        
        # Extract X, y
        X_train = train[FEATURE_COLUMNS].values
        y_train = train["opened"].values
        X_valid = valid[FEATURE_COLUMNS].values
        y_valid = valid["opened"].values
        X_test = test[FEATURE_COLUMNS].values
        y_test = test["opened"].values
        
        mlflow.log_param("features", FEATURE_COLUMNS)
        mlflow.log_param("target", "opened")
        
        # Log target distribution
        train_pos_rate = y_train.mean()
        valid_pos_rate = y_valid.mean()
        mlflow.log_param("train_positive_rate", f"{train_pos_rate:.4f}")
        mlflow.log_param("valid_positive_rate", f"{valid_pos_rate:.4f}")
        print(f"\n  Target distribution:")
        print(f"    Train positive rate: {train_pos_rate:.4f}")
        print(f"    Valid positive rate: {valid_pos_rate:.4f}")
        
        # ========== Step 3: Build Pipeline and Train ==========
        print(f"\n[Step 3] Building and training {args.model.upper()} pipeline...")
        
        if args.model == "logreg":
            # Use hyperparams from config if provided, otherwise use defaults
            if hasattr(args, 'hyperparams') and args.hyperparams:
                hyperparams = args.hyperparams
                print("  Using hyperparams from config file")
            else:
                raise ValueError("Hyperparams are required for Logistic Regression")
            
            mlflow.log_param("hyperparams", hyperparams)
            
            pipeline = build_logreg_pipeline(hyperparams)
            results = train_logreg(
                pipeline, X_train, y_train, X_valid, y_valid, 
                exp_folder, FEATURE_COLUMNS
            )
            
            # Log the entire pipeline
            mlflow.sklearn.log_model(pipeline, "model_pipeline")
            
        elif args.model == "lgbm":
            # Use hyperparams from config if provided, otherwise use defaults
            if hasattr(args, 'hyperparams') and args.hyperparams:
                hyperparams = args.hyperparams
                print("  Using hyperparams from config file")
            else:
                raise ValueError("Hyperparams are required for LightGBM")
            
            mlflow.log_param("hyperparams", hyperparams)
            
            pipeline = build_lgbm_pipeline(hyperparams)
            results = train_lgbm(
                pipeline, X_train, y_train, X_valid, y_valid,
                exp_folder, FEATURE_COLUMNS, hyperparams
            )
            
            # Log the pipeline (note: LightGBM may need special handling)
            mlflow.sklearn.log_model(pipeline, "model_pipeline")
        
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        # ========== Step 4: Confusion Matrix Splits ==========
        print("\n[Step 4] Exporting confusion matrix splits for qualitative analysis...")
        cm_splits = export_confusion_matrix_splits(
            y_valid, results["y_valid_pred"], valid, exp_folder, "valid"
        )
        for category, path in cm_splits.items():
            mlflow.log_artifact(path, artifact_path="confusion_matrix_analysis")
        
        # ========== Step 5: Save Experiment Summary ==========
        print("\n[Step 5] Saving experiment summary...")
        experiment_summary = {
            "experiment_name": args.experiment,
            "run_name": run_name,
            "timestamp": timestamp,
            "model_type": args.model,
            "data_path": args.data_path,
            "features": FEATURE_COLUMNS,
            "train_size": len(train),
            "valid_size": len(valid),
            "test_size": len(test),
            "train_metrics": results["train_metrics"],
            "valid_metrics": results["valid_metrics"],
            "dvc_info": dvc_info
        }
        
        summary_path = f"{exp_folder}/experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        mlflow.log_artifact(summary_path)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"\nResults saved to: {exp_folder}")
        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {results['valid_metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['valid_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['valid_metrics']['recall']:.4f}")
        print(f"  F1 Score:  {results['valid_metrics']['f1']:.4f}")
        
        return experiment_summary


# ============ CLI Entry Point ============

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified ML Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (required). Overrides other CLI args."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["logreg", "lgbm"],
        default=None,
        help="Model type to train: 'logreg' (Logistic Regression) or 'lgbm' (LightGBM)"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/training_data_features.csv",
        help="Path to feature-engineered training data CSV"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (defaults to 'notification_{model}')"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training"
    )
    
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.2,
        help="Proportion of data for validation"
    )
    
    args = parser.parse_args()
    
    # If config file provided, load it and override args
    if args.config:
        config = load_config(args.config)
        args.model = config['model']['name']
        args.hyperparams = config['model']['hyperparams']
        args.train_ratio = config['training']['train_ratio']
        args.valid_ratio = config['training']['valid_ratio']
        args.data_path = config.get('data', {}).get('path', args.data_path)
        args.experiment = config['experiment']['name']
        args.config_path = args.config  # Store for logging
    else:
        # Validate that model is provided if no config
        if args.model is None:
            parser.error("--model is required when not using --config")
        args.hyperparams = None  # Will use defaults in run_training
        args.config_path = None
        
        # Set default experiment name based on model
        if args.experiment is None:
            args.experiment = f"notification_{args.model}"
    
    return args


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
