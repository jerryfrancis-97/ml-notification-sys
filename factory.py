"""
Factory for creating the appropriate training pipeline based on model type.
"""

from __future__ import annotations

from pipelines import LightGBMPipeline, LogisticRegressionPipeline, TrainingPipeline, XGBoostPipeline
from strategies import LightGBMStrategy, LogisticRegressionStrategy, XGBoostStrategy


class TrainerFactory:
    """Factory for creating training pipelines."""

    @staticmethod
    def create_pipeline(model_type: str) -> TrainingPipeline:
        """
        Create the appropriate training pipeline based on model type.

        Args:
            model_type: One of 'logreg', 'lgbm', 'xgb' (and optionally 'xgboost')

        Returns:
            TrainingPipeline: The appropriate pipeline instance
        """
        normalized = (model_type or "").lower()
        if normalized == "xgboost":
            normalized = "xgb"

        strategies = {
            "logreg": LogisticRegressionStrategy(),
            "lgbm": LightGBMStrategy(),
            "xgb": XGBoostStrategy(),
        }

        if normalized not in strategies:
            raise ValueError(f"Unknown model type: {model_type}. Supported: {list(strategies.keys()) + ['xgboost']}")

        pipelines = {
            "logreg": LogisticRegressionPipeline,
            "lgbm": LightGBMPipeline,
            "xgb": XGBoostPipeline,
        }

        strategy = strategies[normalized]
        pipeline_class = pipelines[normalized]
        return pipeline_class(strategy)

