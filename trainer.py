"""
Unified Trainer entry point and CLI.

This module is a thin wrapper around the refactored training components:

- Configuration loading (`config_utils.load_config`)
- Trainer factory (`factory.TrainerFactory`)
- Pipelines / strategies / transformers defined in their own modules

Usage:
    python trainer.py --config configs/logreg.yaml
    python trainer.py --config configs/lgbm.yaml
    python trainer.py --config configs/xgb.yaml
"""

import argparse
import sys
import warnings

from dotenv import load_dotenv

from config_utils import load_config
from factory import TrainerFactory


warnings.filterwarnings("ignore")
load_dotenv(".env")


def run_training(args):
    """
    Main training function to create the appropriate training pipeline and execute it.
    """
    pipeline = TrainerFactory.create_pipeline(args.model)
    return pipeline.run_training(args)


def parse_args():
    """Parse command line arguments for the unified training CLI."""
    parser = argparse.ArgumentParser(
        description="Unified ML Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (required). Overrides other CLI args.",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["logreg", "lgbm", "xgb", "xgboost"],
        default=None,
        help=(
            "Model type to train: 'logreg' (Logistic Regression), "
            "'lgbm' (LightGBM), or 'xgb' / 'xgboost' (XGBoost). "
            "Only used when not providing --config."
        ),
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/training_data_features.csv",
        help="Path to feature-engineered training data CSV",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (defaults to 'notification_{model}')",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training",
    )

    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.2,
        help="Proportion of data for validation",
    )

    args = parser.parse_args()

    # If config file provided, load it and override args
    if args.config:
        config = load_config(args.config)
        args.model = config["model"]["name"]
        args.hyperparams = config["model"]["hyperparams"]
        args.train_ratio = config["training"]["train_ratio"]
        args.valid_ratio = config["training"]["valid_ratio"]
        args.data_path = config.get("data", {}).get("path", args.data_path)
        args.experiment = config["experiment"]["name"]
        args.config_path = args.config  # Store for logging
    else:
        # Validate that model is provided if no config
        if args.model is None:
            parser.error("--model is required when not using --config")

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(0 if run_training(cli_args) is not None else 1)
