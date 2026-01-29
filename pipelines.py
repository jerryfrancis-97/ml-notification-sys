"""
Training pipeline workflow implementations (template method pattern).

This module coordinates:
- Data loading and time-based splitting
- Strategy-specific pipeline build/train
- Artifact generation and MLflow logging
- SHAP analysis (including waterfall plots)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from analysis_utils import export_confusion_matrix_splits
from config_utils import FEATURE_COLUMNS, IMPUTE_COLUMNS, detect_feature_columns
from data_utils import get_dvc_hash, time_based_split
from strategies import (
    LightGBMStrategy,
    LogisticRegressionStrategy,
    TrainingStrategy,
    XGBoostStrategy,
)


plt.switch_backend("Agg")  # Non-interactive backend for SHAP plots


class TrainingPipeline(ABC):
    """Template method pattern for the training pipeline workflow."""

    def __init__(self, strategy: TrainingStrategy):
        self.strategy = strategy

    def run_training(self, args):
        """
        Template method defining the training workflow.

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
            # Create experiment folder first for logging
            exp_folder = f"experiments/{args.experiment}/{timestamp}"
            os.makedirs(exp_folder, exist_ok=True)
            mlflow.log_param("experiment_folder", exp_folder)

            # Configure logging to write to file (not console)
            log_file = os.path.join(exp_folder, "training_log.txt")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8")],
                force=True,  # Override any existing configuration
            )
            logger = logging.getLogger(__name__)

            # Redirect stdout to log file so all print statements are captured
            original_stdout = sys.stdout
            log_file_handle = open(log_file, "w", encoding="utf-8", buffering=1)  # Line buffered
            sys.stdout = log_file_handle

            try:
                print("=" * 60)
                print("Starting Training Pipeline")
                print(f"Model: {str(args.model).upper()}")
                print(f"Experiment: {args.experiment}")
                print(f"Run: {run_name}")
                print("=" * 60)
                print(f"Log file: {log_file}")
                print()

                # Log parameters
                mlflow.log_param("model_type", args.model)
                mlflow.log_param("data_path", args.data_path)
                mlflow.log_param("train_ratio", args.train_ratio)
                mlflow.log_param("valid_ratio", args.valid_ratio)

                # Log DVC hash for data lineage
                dvc_info = get_dvc_hash(args.data_path)
                if dvc_info:
                    mlflow.log_param("data_dvc_md5", dvc_info["md5"])
                    mlflow.log_param("data_dvc_size", dvc_info["size"])
                    logger.info(f"DVC Data Hash: {dvc_info['md5']}")
                    logger.info(f"DVC Data Size: {dvc_info['size']}")

                # Log config file if used
                if hasattr(args, "config_path") and args.config_path:
                    mlflow.log_artifact(args.config_path, artifact_path="config")
                    mlflow.log_param("config_file", args.config_path)
                    logger.info(f"Config file: {args.config_path}")

                # Step 1: Load and prepare data
                data_splits = self._load_and_split_data(args, exp_folder)

                # Step 2: Build and train pipeline
                pipeline = self._build_pipeline(args)
                results = self._train_model(pipeline, data_splits, exp_folder, args)

                # Step 3: Generate additional artifacts
                self._generate_additional_artifacts(results, data_splits, exp_folder)

                # Step 3.5: Perform SHAP analysis
                self._perform_shap_analysis(results, data_splits, exp_folder)

                # Step 4: Save experiment summary
                experiment_summary = self._save_experiment_summary(
                    args, data_splits, results, timestamp, dvc_info, exp_folder
                )

                print("\n" + "=" * 60)
                print("Training Complete!")
                print("=" * 60)
                print(f"\nResults saved to: {exp_folder}")
                print("\nValidation Metrics:")
                print(f"  Accuracy:  {results['valid_metrics']['accuracy']:.4f}")
                print(f"  Precision: {results['valid_metrics']['precision']:.4f}")
                print(f"  Recall:    {results['valid_metrics']['recall']:.4f}")
                print(f"  F1 Score:  {results['valid_metrics']['f1']:.4f}")

            finally:
                # Flush and close log file before logging as artifact
                log_file_handle.flush()
                sys.stdout = original_stdout
                log_file_handle.close()

                # Log the training log file as an artifact
                mlflow.log_artifact(log_file, artifact_path="logs")
                print(f"\nTraining logs saved to: {log_file}")

            return experiment_summary

    def _load_and_split_data(self, args, exp_folder: str) -> Dict[str, Any]:
        """Common data loading and splitting logic."""
        print("\n[Step 1] Loading feature-engineered data...")
        df = pd.read_csv(args.data_path)
        print(f"  Loaded {len(df)} rows from {args.data_path}")
        print(f"  Columns: {list(df.columns)}")

        # Log dataset as artifact
        mlflow.log_artifact(args.data_path, artifact_path="datasets")

        # Sort by timestamp for proper time-based splitting (if timestamp column exists)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        feature_columns, impute_columns = detect_feature_columns(df)
        print(f"  Detected {len(feature_columns)} feature columns: {feature_columns}")
        print(f"  Columns needing imputation: {impute_columns}")

        missing = df[feature_columns].isnull().sum()
        print("\n  Missing values in features (will be imputed):")
        for col, count in missing.items():
            if count > 0:
                print(f"    {col}: {count} ({100 * count / len(df):.2f}%)")

        # Time-based split (BEFORE imputation)
        print("\n[Step 2] Performing time-based split (BEFORE imputation)...")
        train, valid, test = time_based_split(df, args.train_ratio, args.valid_ratio)

        print(f"  Train: {len(train)} rows ({100 * len(train) / len(df):.1f}%)")
        print(f"  Valid: {len(valid)} rows ({100 * len(valid) / len(df):.1f}%)")
        print(f"  Test:  {len(test)} rows ({100 * len(test) / len(df):.1f}%)")

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

        X_train = train[feature_columns].values
        y_train = train["opened"].values
        X_valid = valid[feature_columns].values
        y_valid = valid["opened"].values
        X_test = test[feature_columns].values
        y_test = test["opened"].values

        mlflow.log_param("features", feature_columns)
        mlflow.log_param("target", "opened")

        # Log target distribution
        train_pos_rate = float(y_train.mean())
        valid_pos_rate = float(y_valid.mean())
        mlflow.log_param("train_ratio_of_positive_class", f"{train_pos_rate:.4f}")
        mlflow.log_param("valid_ratio_of_positive_class", f"{valid_pos_rate:.4f}")
        print("\n  Ratio of positive class in target:")
        print(f"    Train ratio of positive class: {train_pos_rate:.4f}")
        print(f"    Valid ratio of positive class: {valid_pos_rate:.4f}")

        return {
            "train": (X_train, y_train),
            "valid": (X_valid, y_valid),
            "test": (X_test, y_test),
            "train_df": train,
            "valid_df": valid,
            "test_df": test,
            "feature_columns": feature_columns,
            "impute_columns": impute_columns,
        }

    @abstractmethod
    def _build_pipeline(self, args):
        """Abstract method: Build the pipeline for this strategy."""

    @abstractmethod
    def _train_model(self, pipeline, data_splits, exp_folder: str, args):
        """Abstract method: Train the model using the strategy."""

    def _generate_additional_artifacts(self, results: Dict[str, Any], data_splits: Dict[str, Any], exp_folder: str):
        """Common additional artifact generation."""
        print("\n[Step 4] Exporting confusion matrix splits for qualitative analysis...")

        # Use imputed validation data for confusion matrix analysis (matches what model saw)
        imputed_valid_df = results.get("X_valid_imputed_df", data_splits["valid_df"])

        cm_splits = export_confusion_matrix_splits(
            data_splits["valid"][1], results["y_valid_pred"], imputed_valid_df, exp_folder, "valid"
        )
        for _, path in cm_splits.items():
            mlflow.log_artifact(path, artifact_path="confusion_matrix_analysis")

    def _get_training_sample(self, X_train_data, sample_size: int = 500):
        """Sample from training data only for SHAP explainer background."""
        if sample_size >= len(X_train_data):
            return X_train_data
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_train_data), size=sample_size, replace=False)
        return X_train_data[sample_indices]

    def _generate_waterfall_plots(
        self,
        explainer,
        X_valid_data,
        shap_values_valid,
        y_valid,
        y_valid_pred,
        feature_columns,
        exp_folder: str,
    ):
        """Generate waterfall plots for samples from each confusion matrix class."""
        print("Generating waterfall plots for confusion matrix classes...")

        # Identify classes
        tp_mask = (y_valid == 1) & (y_valid_pred == 1)
        tn_mask = (y_valid == 0) & (y_valid_pred == 0)
        fp_mask = (y_valid == 0) & (y_valid_pred == 1)
        fn_mask = (y_valid == 1) & (y_valid_pred == 0)

        classes = {
            "TP": (tp_mask, "True Positive"),
            "TN": (tn_mask, "True Negative"),
            "FP": (fp_mask, "False Positive"),
            "FN": (fn_mask, "False Negative"),
        }

        np.random.seed(42)
        samples_per_class = 2  # Number of samples to plot per class

        waterfall_files = []

        for class_name, (mask, label) in classes.items():
            indices = np.where(mask)[0]
            if len(indices) == 0:
                print(f"  No samples found for {label}, skipping...")
                continue

            n_samples = min(samples_per_class, len(indices))
            selected_indices = np.random.choice(indices, size=n_samples, replace=False)

            for i, idx in enumerate(selected_indices):
                shap_values_instance = shap_values_valid[idx]
                X_instance = X_valid_data[idx : idx + 1]

                # Get expected value (base value)
                if hasattr(explainer, "expected_value"):
                    base_value = explainer.expected_value
                    if isinstance(base_value, np.ndarray):
                        if len(base_value) > 1:
                            base_value = base_value[1]  # For binary classification, use positive class
                        else:
                            base_value = base_value[0]
                else:
                    base_value = 0.0

                explanation = shap.Explanation(
                    values=shap_values_instance.reshape(1, -1),
                    base_values=np.array([base_value]),
                    data=X_instance,
                    feature_names=feature_columns,
                )

                plt.figure(figsize=(10, 8))
                try:
                    shap.plots.waterfall(explanation[0], show=False, max_display=20)
                except Exception as e:
                    print(f"    Warning: Waterfall plot failed for {label} sample {i + 1}, using bar plot: {e}")
                    shap.plots.bar(explanation[0], show=False, max_display=20)
                plt.title(f"{label} - Sample {i + 1} (True: {y_valid[idx]}, Pred: {y_valid_pred[idx]})")
                plt.tight_layout()
                filename = f"{exp_folder}/shap_waterfall_{class_name}_{i}.png"
                plt.savefig(filename, dpi=150, bbox_inches="tight")
                plt.close()
                waterfall_files.append(filename)
                print(f"  Saved {label} sample {i + 1}: {filename}")

        return waterfall_files

    def _perform_shap_analysis(self, results: Dict[str, Any], data_splits: Dict[str, Any], exp_folder: str):
        """Perform comprehensive SHAP analysis on validation data."""
        print("\n[Step 3.5] Performing SHAP analysis...")

        pipeline = results.get("pipeline")
        if pipeline is None:
            print("  WARNING: No pipeline found in results, skipping SHAP analysis")
            return

        feature_columns = data_splits.get("feature_columns", [])
        X_train, y_train = data_splits["train"]
        X_valid, y_valid = data_splits["valid"]
        y_valid_pred = results.get("y_valid_pred")

        if y_valid_pred is None:
            print("  WARNING: No validation predictions found, skipping SHAP analysis")
            return

        # Extract pipeline components
        imputer = pipeline.named_steps["imputer"]
        has_scaler = "scaler" in pipeline.named_steps

        # Transform data based on model type
        X_train_imputed = imputer.transform(X_train)
        X_valid_imputed = imputer.transform(X_valid)

        if has_scaler:
            scaler = pipeline.named_steps["scaler"]
            X_train_for_shap = scaler.transform(X_train_imputed)
            X_valid_for_shap = scaler.transform(X_valid_imputed)
        else:
            X_train_for_shap = X_train_imputed
            X_valid_for_shap = X_valid_imputed

        background_sample = self._get_training_sample(X_train_for_shap, sample_size=500)
        print(f"  Using {background_sample.shape} training samples for SHAP explainer background")

        # Get classifier from the strategy instance (LightGBM/XGBoost) or pipeline (LogisticRegression)
        if hasattr(self.strategy, "classifier"):
            classifier = self.strategy.classifier
        else:
            classifier = pipeline.named_steps["classifier"]

        if isinstance(classifier, (LGBMClassifier, XGBClassifier)):
            try:
                explainer = shap.TreeExplainer(classifier, data=background_sample)
            except Exception:
                explainer = shap.TreeExplainer(classifier, data=background_sample)
        else:
            explainer = shap.LinearExplainer(classifier, background_sample)

        print(f"  Calculating SHAP values on {len(X_valid_for_shap)} validation samples...")
        shap_values_valid = explainer.shap_values(X_valid_for_shap)

        if isinstance(shap_values_valid, list) and len(shap_values_valid) == 2:
            shap_values_valid = shap_values_valid[1]

        print("  Generating SHAP plots...")

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_valid, X_valid_for_shap, feature_names=feature_columns, show=False, max_display=20)
        plt.tight_layout()
        summary_plot_path = f"{exp_folder}/shap_summary_plot.png"
        plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_valid,
            X_valid_for_shap,
            feature_names=feature_columns,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        importance_plot_path = f"{exp_folder}/shap_feature_importance.png"
        plt.savefig(importance_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        mean_abs_shap = np.abs(shap_values_valid).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-5:][::-1]

        dependence_files = []
        for feature_idx in top_indices:
            feature_name = feature_columns[feature_idx]
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feature_idx, shap_values_valid, X_valid_for_shap, feature_names=feature_columns, show=False)
            plt.tight_layout()
            dependence_path = f"{exp_folder}/shap_dependence_{feature_name}.png"
            plt.savefig(dependence_path, dpi=150, bbox_inches="tight")
            plt.close()
            dependence_files.append(dependence_path)

        waterfall_files = self._generate_waterfall_plots(
            explainer, X_valid_for_shap, shap_values_valid, y_valid, y_valid_pred, feature_columns, exp_folder
        )

        mean_abs_shap = np.abs(shap_values_valid).mean(axis=0)
        std_abs_shap = np.abs(shap_values_valid).std(axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_abs_shap": mean_abs_shap,
                "std_abs_shap": std_abs_shap,
                "importance_rank": np.argsort(mean_abs_shap)[::-1] + 1,
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_csv_path = f"{exp_folder}/shap_feature_importance.csv"
        importance_df.to_csv(importance_csv_path, index=False)

        shap_values_path = f"{exp_folder}/shap_values.npy"
        np.save(shap_values_path, shap_values_valid)

        X_valid_df = pd.DataFrame(X_valid_for_shap, columns=feature_columns)
        sample_data_path = f"{exp_folder}/shap_sample_data.csv"
        X_valid_df.to_csv(sample_data_path, index=False)

        summary = {
            "num_features": len(feature_columns),
            "num_validation_samples": len(X_valid_for_shap),
            "background_sample_size": len(background_sample),
            "top_feature": importance_df.iloc[0]["feature"] if len(importance_df) > 0 else None,
            "explainer_type": "TreeExplainer" if isinstance(classifier, (LGBMClassifier, XGBClassifier)) else "LinearExplainer",
            "analysis_timestamp": datetime.now().isoformat(),
        }

        summary_path = f"{exp_folder}/shap_analysis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print("  Logging SHAP artifacts to MLflow...")
        artifacts_to_log = [
            summary_plot_path,
            importance_plot_path,
            importance_csv_path,
            shap_values_path,
            sample_data_path,
            summary_path,
        ]
        artifacts_to_log.extend(dependence_files)
        artifacts_to_log.extend(waterfall_files)

        for artifact_path in artifacts_to_log:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path, artifact_path="shap_analysis")

        print("  SHAP analysis completed!")

    def _save_experiment_summary(
        self,
        args,
        data_splits: Dict[str, Any],
        results: Dict[str, Any],
        timestamp: str,
        dvc_info: Optional[Dict[str, Any]],
        exp_folder: str,
    ) -> Dict[str, Any]:
        """Common experiment summary saving."""
        print("\n[Step 5] Saving experiment summary...")

        pipeline = results.get("pipeline")
        if pipeline:
            imputer = pipeline.named_steps["imputer"]
            feature_columns = data_splits.get("feature_columns", FEATURE_COLUMNS)
            imputation_stats = imputer.get_imputation_report(feature_columns)

            stats_path = f"{exp_folder}/imputation_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(imputation_stats, f, indent=2)
            mlflow.log_artifact(stats_path)
            print(f"  Saved imputation statistics to: {stats_path}")
        else:
            imputation_stats = None

        experiment_summary = {
            "experiment_name": args.experiment,
            "run_name": f"{args.model}_run_{timestamp}",
            "timestamp": timestamp,
            "model_type": args.model,
            "data_path": args.data_path,
            "features": data_splits.get("feature_columns", FEATURE_COLUMNS),
            "impute_columns": data_splits.get("impute_columns", IMPUTE_COLUMNS),
            "train_size": len(data_splits["train"][0]),
            "valid_size": len(data_splits["valid"][0]),
            "test_size": len(data_splits["test"][0]),
            "train_metrics": results["train_metrics"],
            "valid_metrics": results["valid_metrics"],
            "imputation_stats": imputation_stats,
            "dvc_info": dvc_info,
        }

        summary_path = f"{exp_folder}/experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(experiment_summary, f, indent=2)
        mlflow.log_artifact(summary_path)

        return experiment_summary


class LogisticRegressionPipeline(TrainingPipeline):
    """Concrete implementation for Logistic Regression training."""

    def _build_pipeline(self, args):
        print("\n[Step 3] Building Logistic Regression pipeline...")
        if not hasattr(args, "hyperparams") or not args.hyperparams:
            raise ValueError("Hyperparams are required for Logistic Regression")

        hyperparams = args.hyperparams
        if hyperparams.get("penalty") == "None":
            hyperparams["penalty"] = None

        mlflow.log_param("hyperparams", hyperparams)
        return self.strategy.build_pipeline(hyperparams)

    def _train_model(self, pipeline, data_splits, exp_folder: str, args):
        X_train, y_train = data_splits["train"]
        X_valid, y_valid = data_splits["valid"]

        feature_columns = data_splits.get("feature_columns", FEATURE_COLUMNS)

        results = self.strategy.train(
            pipeline, X_train, y_train, X_valid, y_valid, exp_folder, feature_columns, args.hyperparams, data_splits
        )

        mlflow.sklearn.log_model(sk_model=pipeline, name="model_pipeline")
        results["pipeline"] = pipeline
        return results


class LightGBMPipeline(TrainingPipeline):
    """Concrete implementation for LightGBM training."""

    def _build_pipeline(self, args):
        print("\n[Step 3] Building LightGBM pipeline...")
        if not hasattr(args, "hyperparams") or not args.hyperparams:
            raise ValueError("Hyperparams are required for LightGBM")

        mlflow.log_param("hyperparams", args.hyperparams)
        return self.strategy.build_pipeline(args.hyperparams)

    def _train_model(self, pipeline, data_splits, exp_folder: str, args):
        X_train, y_train = data_splits["train"]
        X_valid, y_valid = data_splits["valid"]

        feature_columns = data_splits.get("feature_columns", FEATURE_COLUMNS)

        results = self.strategy.train(
            pipeline, X_train, y_train, X_valid, y_valid, exp_folder, feature_columns, args.hyperparams, data_splits
        )

        mlflow.sklearn.log_model(sk_model=pipeline, name="model_pipeline")
        results["pipeline"] = pipeline
        return results


class XGBoostPipeline(TrainingPipeline):
    """Concrete implementation for XGBoost training."""

    def _build_pipeline(self, args):
        print("\n[Step 3] Building XGBoost pipeline...")
        if not hasattr(args, "hyperparams") or not args.hyperparams:
            raise ValueError("Hyperparams are required for XGBoost")

        mlflow.log_param("hyperparams", args.hyperparams)
        return self.strategy.build_pipeline(args.hyperparams)

    def _train_model(self, pipeline, data_splits, exp_folder: str, args):
        X_train, y_train = data_splits["train"]
        X_valid, y_valid = data_splits["valid"]

        feature_columns = data_splits.get("feature_columns", FEATURE_COLUMNS)

        results = self.strategy.train(
            pipeline, X_train, y_train, X_valid, y_valid, exp_folder, feature_columns, args.hyperparams, data_splits
        )

        mlflow.sklearn.log_model(sk_model=pipeline, name="model_pipeline")
        results["pipeline"] = pipeline
        return results


# Re-export strategies/pipelines for consumers that follow the old import chain.
__all__ = [
    "TrainingPipeline",
    "LogisticRegressionPipeline",
    "LightGBMPipeline",
    "XGBoostPipeline",
    "TrainingStrategy",
    "LogisticRegressionStrategy",
    "LightGBMStrategy",
    "XGBoostStrategy",
]

