"""
Model callback helpers.

Currently includes LightGBM callbacks used by the training strategies.
"""

from __future__ import annotations

from typing import Any, Dict


def lgb_early_stopping_callback(stopping_rounds: int):
    """Create LightGBM early stopping callback."""
    from lightgbm import early_stopping

    return early_stopping(stopping_rounds=stopping_rounds, verbose=True)


def lgb_record_evaluation_callback(evals_result: Dict[str, Dict[str, Any]]):
    """Create LightGBM callback to record evaluation results."""
    from lightgbm import record_evaluation

    return record_evaluation(evals_result)

