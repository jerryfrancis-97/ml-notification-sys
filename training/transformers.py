"""
Custom sklearn transformer utilities used in the training pipelines.

Currently contains:
- FeatureImputerTransformer: wraps SimpleImputer to avoid data leakage and
  provide convenient reporting of imputation statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class FeatureImputerTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible imputer that:
    - fit(): Learns statistics (mean) from training data ONLY
    - transform(): Applies those SAME training statistics to any data

    This prevents data leakage by ensuring validation/test sets use
    only training set statistics for imputation.
    """

    def __init__(self, strategy: str = "mean") -> None:
        self.strategy = strategy
        self.imputer_: Optional[SimpleImputer] = None
        self.statistics_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureImputerTransformer":
        """Learn imputation statistics from training data ONLY."""
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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply TRAINING statistics to transform any data."""
        if self.imputer_ is None:
            raise ValueError("Imputer not fitted. Call fit() first.")

        X_transformed = self.imputer_.transform(X)
        return X_transformed

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> Optional[List[str]]:
        """Return feature names for sklearn compatibility."""
        return input_features

    def get_imputation_report(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return detailed imputation statistics for logging."""
        if self.statistics_ is None:
            raise ValueError("Imputer not fitted. Call fit() first.")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.statistics_))]

        return {
            "strategy": self.strategy,
            "statistics": dict(zip(feature_names, self.statistics_)),
            "features_with_missing": [
                name for name, stat in zip(feature_names, self.statistics_) if not np.isnan(stat)
            ],
        }

