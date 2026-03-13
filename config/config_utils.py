"""
Configuration and feature-detection utilities shared across training modules.

This module centralizes:
- Static feature column definitions used for training
- Dynamic feature / imputation column detection from a DataFrame
- YAML-based config loading for the training CLI
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml


def detect_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Dynamically detect feature columns and columns needing imputation from loaded data.

    Args:
        df: Loaded DataFrame from CSV

    Returns:
        tuple: (feature_columns, impute_columns)
    """
    exclude_cols = {
        "user_id", "opened", "timestamp", "time_bucket", "day", "hour",
        "send_timestamp", "open_timestamp", "response_delay_minutes",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_cols if col not in exclude_cols]

    impute_columns: List[str] = []
    for col in feature_columns:
        if df[col].isnull().any():
            impute_columns.append(col)

    return feature_columns, impute_columns


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"\n[Config] Loaded configuration from: {config_path}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Experiment: {config['experiment']['name']}")

    return config


# ============ Feature Columns ============

FEATURE_COLUMNS: List[str] = [
    "num_notifications_last_24h",
    "delay_since_last_open_notification",
    "user_open_rate",
    "user_hour_open_rate",
    "user_morning_open_rate",
    "user_afternoon_open_rate",
    "user_evening_open_rate",
    "user_night_open_rate",
    "user_fatigue_ratio",
    "hour_sin",
    "hour_cos",
    "hour_x_user_open_rate",
    "hour_x_user_hour_open_rate",
    "hour_x_num_notifications_last_24h",
    "hour_x_delay_since_last_open_notification",
]

IMPUTE_COLUMNS: List[str] = [
    "user_open_rate",
    "user_hour_open_rate",
    "user_morning_open_rate",
    "user_afternoon_open_rate",
    "user_evening_open_rate",
    "user_night_open_rate",
    "user_fatigue_ratio",
]

