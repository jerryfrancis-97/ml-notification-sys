"""
Training strategy implementations.

Each strategy is responsible for:
- Building an sklearn Pipeline suitable for the model
- Training the model (handling early stopping / eval sets when needed)
- Generating model-specific artifacts and MLflow logging
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from callbacks import lgb_early_stopping_callback, lgb_record_evaluation_callback
from data_utils import evaluate_model
from transformers import FeatureImputerTransformer
from viz_utils import (
    compute_metrics_per_round,
    plot_accuracy_curves,
    plot_coefficients,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_f1_curves,
    plot_learning_curve,
    plot_loss_learning_curve,
    plot_loss_learning_curve_lgbm,
    plot_pr_curve,
    plot_precision_curves,
    plot_recall_curves,
    plot_roc_curve,
    plot_threshold_vs_pr,
    plot_training_history,
)


class TrainingStrategy(ABC):
    """Abstract base class for different training strategies."""

    @abstractmethod
    def build_pipeline(self, hyperparams: Dict[str, Any]) -> Pipeline:
        """Build the sklearn pipeline for this strategy."""

    @abstractmethod
    def train(
        self,
        pipeline: Pipeline,
        X_train,
        y_train,
        X_valid,
        y_valid,
        exp_folder: str,
        features,
        hyperparams: Dict[str, Any],
        data_splits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train the model and generate all artifacts."""


class LogisticRegressionStrategy(TrainingStrategy):
    """Strategy for training Logistic Regression models."""

    def build_pipeline(self, hyperparams: Dict[str, Any]) -> Pipeline:
        """Build sklearn Pipeline for Logistic Regression."""
        pipeline = Pipeline(
            [
                ("imputer", FeatureImputerTransformer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**hyperparams)),
            ]
        )
        return pipeline

    def train(
        self,
        pipeline: Pipeline,
        X_train,
        y_train,
        X_valid,
        y_valid,
        exp_folder: str,
        features,
        hyperparams: Dict[str, Any],
        data_splits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train LogisticRegression pipeline and generate all artifacts."""
        print("\n" + "=" * 50)
        print("Training Logistic Regression Pipeline")
        print("=" * 50)

        # Fit pipeline (imputer + scaler + classifier fitted on train only)
        pipeline.fit(X_train, y_train)

        # Get the fitted classifier for additional analysis
        classifier = pipeline.named_steps["classifier"]
        scaler = pipeline.named_steps["scaler"]
        imputer = pipeline.named_steps["imputer"]

        # Get transformed data for learning curves (need to transform manually)
        X_train_imputed = imputer.transform(X_train)
        X_train_scaled = scaler.transform(X_train_imputed)

        # Get imputed validation data for confusion matrix analysis
        X_valid_imputed = imputer.transform(X_valid)

        # Create imputed validation DataFrame for confusion matrix analysis
        X_valid_imputed_df = data_splits["valid_df"].copy()
        X_valid_imputed_df[features] = X_valid_imputed

        # Predictions using custom threshold
        y_train_pred = pipeline.predict(X_train)
        y_valid_pred = pipeline.predict(X_valid)
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
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
        plot_confusion_matrix(
            y_train, y_train_pred, f"{exp_folder}/confusion_matrix_train.png", "Training"
        )
        plot_confusion_matrix(
            y_valid, y_valid_pred, f"{exp_folder}/confusion_matrix_valid.png", "Validation"
        )
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_train.png")
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_valid.png")

        # Learning curves (using scaled training data)
        print("Generating learning curve...")
        plot_learning_curve(classifier, X_train_scaled, y_train, f"{exp_folder}/learning_curve.png", cv=5)
        mlflow.log_artifact(f"{exp_folder}/learning_curve.png")

        print("Generating loss learning curve...")
        plot_loss_learning_curve(
            classifier, X_train_scaled, y_train, f"{exp_folder}/learning_curve_loss.png", cv=5
        )
        mlflow.log_artifact(f"{exp_folder}/learning_curve_loss.png")

        # PR curves
        train_pr_auc = plot_pr_curve(y_train, y_train_proba, f"{exp_folder}/pr_curve_train.png", "Training")
        valid_pr_auc = plot_pr_curve(y_valid, y_valid_proba, f"{exp_folder}/pr_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_train.png")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_valid.png")
        mlflow.log_metric("train_pr_auc", train_pr_auc)
        mlflow.log_metric("valid_pr_auc", valid_pr_auc)

        # ROC curves
        train_roc_auc = plot_roc_curve(y_train, y_train_proba, f"{exp_folder}/roc_curve_train.png", "Training")
        valid_roc_auc = plot_roc_curve(y_valid, y_valid_proba, f"{exp_folder}/roc_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_train.png")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_valid.png")
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("valid_roc_auc", valid_roc_auc)

        # Threshold vs Precision/Recall analysis
        print("Generating threshold vs PR analysis...")
        plot_threshold_vs_pr(y_train, y_train_proba, f"{exp_folder}/threshold_pr_analysis_train.png", "Training")
        plot_threshold_vs_pr(y_valid, y_valid_proba, f"{exp_folder}/threshold_pr_analysis_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/threshold_pr_analysis_train.png")
        mlflow.log_artifact(f"{exp_folder}/threshold_pr_analysis_valid.png")

        # Coefficients
        coef_df = plot_coefficients(classifier, features, f"{exp_folder}/coefficients.png")
        coef_df.to_csv(f"{exp_folder}/coefficients.csv", index=False)
        mlflow.log_artifact(f"{exp_folder}/coefficients.png")
        mlflow.log_artifact(f"{exp_folder}/coefficients.csv")

        return {
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "y_valid_pred": y_valid_pred,
            "y_valid_proba": y_valid_proba,
            "X_valid_imputed_df": X_valid_imputed_df,
        }


class LightGBMStrategy(TrainingStrategy):
    """Strategy for training LightGBM models."""

    def build_pipeline(self, hyperparams: Dict[str, Any]) -> Pipeline:
        """Build sklearn Pipeline for LightGBM."""
        pipeline = Pipeline(
            [
                ("imputer", FeatureImputerTransformer(strategy="mean")),
                ("classifier", LGBMClassifier(**hyperparams)),
            ]
        )
        return pipeline

    def train(
        self,
        pipeline: Pipeline,
        X_train,
        y_train,
        X_valid,
        y_valid,
        exp_folder: str,
        features,
        hyperparams: Dict[str, Any],
        data_splits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train LightGBM pipeline and generate all artifacts."""
        print("\n" + "=" * 50)
        print("Training LightGBM Pipeline")
        print("=" * 50)

        # For LightGBM with early stopping, handle early stopping via callbacks.
        imputer = pipeline.named_steps["imputer"]
        imputer.fit(X_train)

        # Transform both train and valid using TRAINING statistics
        X_train_imputed = imputer.transform(X_train)
        X_valid_imputed = imputer.transform(X_valid)

        print(f"\nImputed training data shape: {X_train_imputed.shape}")
        print(f"Imputed validation data shape: {X_valid_imputed.shape}")

        # Create imputed validation DataFrame for confusion matrix analysis
        X_valid_imputed_df = data_splits["valid_df"].copy()
        X_valid_imputed_df[features] = X_valid_imputed

        # Get hyperparams and handle early stopping
        lgbm_params = hyperparams.copy()
        early_stopping_rounds = lgbm_params.pop("early_stopping_rounds", 50)

        classifier = LGBMClassifier(**lgbm_params)

        # Store evaluation results
        evals_result: Dict[str, Dict[str, Any]] = {}

        classifier = classifier.fit(
            X_train_imputed,
            y_train,
            eval_set=[(X_train_imputed, y_train), (X_valid_imputed, y_valid)],
            eval_names=["training", "valid_1"],
            eval_metric=["binary_logloss", "binary_error"],
            callbacks=[
                lgb_early_stopping_callback(early_stopping_rounds),
                lgb_record_evaluation_callback(evals_result),
            ],
        )

        # Store classifier as instance variable
        self.classifier = classifier

        # Predictions
        y_train_pred = self.classifier.predict(X_train_imputed)
        y_train_proba = self.classifier.predict_proba(X_train_imputed)[:, 1]
        y_valid_pred = self.classifier.predict(X_valid_imputed)
        y_valid_proba = self.classifier.predict_proba(X_valid_imputed)[:, 1]

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
        mlflow.log_metric("best_iteration", self.classifier.best_iteration_)
        mlflow.log_metric("n_estimators_used", self.classifier.n_estimators_)

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Confusion matrices
        plot_confusion_matrix(
            y_train, y_train_pred, f"{exp_folder}/confusion_matrix_train.png", "Training"
        )
        plot_confusion_matrix(
            y_valid, y_valid_pred, f"{exp_folder}/confusion_matrix_valid.png", "Validation"
        )
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_train.png")
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_valid.png")

        # Training history
        plot_training_history(evals_result, f"{exp_folder}/training_history.png", model_name="LightGBM")
        mlflow.log_artifact(f"{exp_folder}/training_history.png")

        # Accuracy curves
        plot_accuracy_curves(evals_result, f"{exp_folder}/accuracy_curves.png", model_name="LightGBM")
        mlflow.log_artifact(f"{exp_folder}/accuracy_curves.png")

        # Compute per-round metrics for F1, precision, recall curves
        print("Computing per-round metrics...")
        metrics_per_round = compute_metrics_per_round(
            self.classifier, X_train_imputed, y_train, X_valid_imputed, y_valid
        )

        plot_f1_curves(metrics_per_round, f"{exp_folder}/f1_curves.png", model_name="LightGBM")
        plot_precision_curves(metrics_per_round, f"{exp_folder}/precision_curves.png", model_name="LightGBM")
        plot_recall_curves(metrics_per_round, f"{exp_folder}/recall_curves.png", model_name="LightGBM")
        mlflow.log_artifact(f"{exp_folder}/f1_curves.png")
        mlflow.log_artifact(f"{exp_folder}/precision_curves.png")
        mlflow.log_artifact(f"{exp_folder}/recall_curves.png")

        # Loss learning curve
        print("Generating loss learning curve...")
        plot_loss_learning_curve_lgbm(
            X_train_imputed,
            y_train,
            X_valid_imputed,
            y_valid,
            lgbm_params,
            f"{exp_folder}/learning_curve_loss.png",
        )
        mlflow.log_artifact(f"{exp_folder}/learning_curve_loss.png")

        # PR curves
        train_pr_auc = plot_pr_curve(y_train, y_train_proba, f"{exp_folder}/pr_curve_train.png", "Training")
        valid_pr_auc = plot_pr_curve(y_valid, y_valid_proba, f"{exp_folder}/pr_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_train.png")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_valid.png")
        mlflow.log_metric("train_pr_auc", train_pr_auc)
        mlflow.log_metric("valid_pr_auc", valid_pr_auc)

        # ROC curves
        train_roc_auc = plot_roc_curve(y_train, y_train_proba, f"{exp_folder}/roc_curve_train.png", "Training")
        valid_roc_auc = plot_roc_curve(y_valid, y_valid_proba, f"{exp_folder}/roc_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_train.png")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_valid.png")
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("valid_roc_auc", valid_roc_auc)

        # Feature importance
        importance_df = plot_feature_importance(
            self.classifier, features, f"{exp_folder}/feature_importance.png", model_name="LightGBM"
        )
        importance_df.to_csv(f"{exp_folder}/feature_importance.csv", index=False)
        mlflow.log_artifact(f"{exp_folder}/feature_importance.png")
        mlflow.log_artifact(f"{exp_folder}/feature_importance.csv")

        # Close any figures created by downstream plotting utilities.
        plt.close("all")

        return {
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "y_valid_pred": y_valid_pred,
            "y_valid_proba": y_valid_proba,
            "X_valid_imputed_df": X_valid_imputed_df,
        }


class XGBoostStrategy(TrainingStrategy):
    """Strategy for training XGBoost models."""

    def build_pipeline(self, hyperparams: Dict[str, Any]) -> Pipeline:
        """Build sklearn Pipeline for XGBoost."""
        pipeline = Pipeline(
            [
                ("imputer", FeatureImputerTransformer(strategy="mean")),
                ("classifier", XGBClassifier(**hyperparams)),
            ]
        )
        return pipeline

    def train(
        self,
        pipeline: Pipeline,
        X_train,
        y_train,
        X_valid,
        y_valid,
        exp_folder: str,
        features,
        hyperparams: Dict[str, Any],
        data_splits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train XGBoost pipeline and generate all artifacts."""
        print("\n" + "=" * 50)
        print("Training XGBoost Pipeline")
        print("=" * 50)

        # We cannot use pipeline.fit(X_train, y_train) because XGBClassifier.fit()
        # requires eval_set. sklearn Pipeline.fit(X, y) can't pass that to the final estimator.
        imputer = pipeline.named_steps["imputer"]
        imputer.fit(X_train)

        # Transform both train and valid using TRAINING statistics
        X_train_imputed = imputer.transform(X_train)
        X_valid_imputed = imputer.transform(X_valid)

        print(f"\nImputed training data shape: {X_train_imputed.shape}")
        print(f"Imputed validation data shape: {X_valid_imputed.shape}")

        # Create imputed validation DataFrame for confusion matrix analysis
        X_valid_imputed_df = data_splits["valid_df"].copy()
        X_valid_imputed_df[features] = X_valid_imputed

        # Get hyperparams and handle early stopping
        xgb_params = hyperparams.copy()
        early_stopping_rounds = xgb_params.pop("early_stopping_rounds", 50)
        xgb_params.setdefault("callbacks", [])
        xgb_params["callbacks"] = [EarlyStopping(rounds=early_stopping_rounds)] + list(xgb_params["callbacks"])

        classifier = XGBClassifier(**xgb_params)

        classifier = classifier.fit(
            X_train_imputed,
            y_train,
            eval_set=[(X_train_imputed, y_train), (X_valid_imputed, y_valid)],
        )

        # Store classifier as instance variable
        self.classifier = classifier

        # Get evaluation results from classifier
        evals_result = self.classifier.evals_result_ if hasattr(self.classifier, "evals_result_") else {}

        # Transform XGBoost evals_result format to match LightGBM format for plotting
        plot_eval_results = {"training": {"binary_logloss": None, "binary_error": None}, "valid_1": {"binary_logloss": None, "binary_error": None}}
        for key in evals_result.keys():
            if key == "validation_0":
                plot_eval_results["training"]["binary_logloss"] = evals_result[key]["logloss"]
                plot_eval_results["training"]["binary_error"] = evals_result[key]["error"]
            elif key == "validation_1":
                plot_eval_results["valid_1"]["binary_logloss"] = evals_result[key]["logloss"]
                plot_eval_results["valid_1"]["binary_error"] = evals_result[key]["error"]
            else:
                raise ValueError(f"Invalid key: {key}")

        # Predictions
        y_train_pred = self.classifier.predict(X_train_imputed)
        y_train_proba = self.classifier.predict_proba(X_train_imputed)[:, 1]
        y_valid_pred = self.classifier.predict(X_valid_imputed)
        y_valid_proba = self.classifier.predict_proba(X_valid_imputed)[:, 1]

        # Evaluate
        train_metrics = evaluate_model(y_train, y_train_pred, "Training")
        valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")

        # Log metrics to MLflow
        for prefix, metrics in [("train", train_metrics), ("valid", valid_metrics)]:
            mlflow.log_metric(f"{prefix}_accuracy", metrics["accuracy"])
            mlflow.log_metric(f"{prefix}_precision", metrics["precision"])
            mlflow.log_metric(f"{prefix}_recall", metrics["recall"])
            mlflow.log_metric(f"{prefix}_f1", metrics["f1"])

        # Log XGBoost specific info
        if hasattr(self.classifier, "best_iteration"):
            mlflow.log_metric("best_iteration", self.classifier.best_iteration)
        mlflow.log_metric("n_estimators_used", self.classifier.n_estimators)

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Confusion matrices
        plot_confusion_matrix(
            y_train, y_train_pred, f"{exp_folder}/confusion_matrix_train.png", "Training"
        )
        plot_confusion_matrix(
            y_valid, y_valid_pred, f"{exp_folder}/confusion_matrix_valid.png", "Validation"
        )
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_train.png")
        mlflow.log_artifact(f"{exp_folder}/confusion_matrix_valid.png")

        # Training history (if available)
        if evals_result:
            plot_training_history(plot_eval_results, f"{exp_folder}/training_history.png", model_name="XGBoost")
            mlflow.log_artifact(f"{exp_folder}/training_history.png")

            # Accuracy curves
            plot_accuracy_curves(plot_eval_results, f"{exp_folder}/accuracy_curves.png", model_name="XGBoost")
            mlflow.log_artifact(f"{exp_folder}/accuracy_curves.png")

        # Compute per-round metrics for F1, precision, recall curves
        print("Computing per-round metrics...")
        metrics_per_round = compute_metrics_per_round(
            self.classifier, X_train_imputed, y_train, X_valid_imputed, y_valid
        )

        plot_f1_curves(metrics_per_round, f"{exp_folder}/f1_curves.png", model_name="XGBoost")
        plot_precision_curves(metrics_per_round, f"{exp_folder}/precision_curves.png", model_name="XGBoost")
        plot_recall_curves(metrics_per_round, f"{exp_folder}/recall_curves.png", model_name="XGBoost")
        mlflow.log_artifact(f"{exp_folder}/f1_curves.png")
        mlflow.log_artifact(f"{exp_folder}/precision_curves.png")
        mlflow.log_artifact(f"{exp_folder}/recall_curves.png")

        # PR curves
        train_pr_auc = plot_pr_curve(y_train, y_train_proba, f"{exp_folder}/pr_curve_train.png", "Training")
        valid_pr_auc = plot_pr_curve(y_valid, y_valid_proba, f"{exp_folder}/pr_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_train.png")
        mlflow.log_artifact(f"{exp_folder}/pr_curve_valid.png")
        mlflow.log_metric("train_pr_auc", train_pr_auc)
        mlflow.log_metric("valid_pr_auc", valid_pr_auc)

        # ROC curves
        train_roc_auc = plot_roc_curve(y_train, y_train_proba, f"{exp_folder}/roc_curve_train.png", "Training")
        valid_roc_auc = plot_roc_curve(y_valid, y_valid_proba, f"{exp_folder}/roc_curve_valid.png", "Validation")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_train.png")
        mlflow.log_artifact(f"{exp_folder}/roc_curve_valid.png")
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("valid_roc_auc", valid_roc_auc)

        # Feature importance
        importance_df = plot_feature_importance(
            self.classifier, features, f"{exp_folder}/feature_importance.png", model_name="XGBoost"
        )
        importance_df.to_csv(f"{exp_folder}/feature_importance.csv", index=False)
        mlflow.log_artifact(f"{exp_folder}/feature_importance.png")
        mlflow.log_artifact(f"{exp_folder}/feature_importance.csv")

        plt.close("all")

        return {
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "y_valid_pred": y_valid_pred,
            "y_valid_proba": y_valid_proba,
            "X_valid_imputed_df": X_valid_imputed_df,
        }

