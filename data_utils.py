"""
Data loading and preparation utilities.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DataLoader:
    """
    Class for loading and preparing notification data for ML experiments.
    Handles data loading, time-based splitting, and feature extraction.
    """
    
    FEATURE_COLUMNS = [
        "hour", "day_of_week", "is_weekend",
        "num_notifications_last_24h", "delay_since_last_open_notification",
        "user_open_rate", "user_hour_open_rate", "hour_sin", "hour_cos"
    ]
    
    def __init__(self, data_path):
        """
        Initialize DataLoader with path to data file.
        
        Args:
            data_path: Path to the CSV file containing the data
        """
        self.data_path = data_path
        self.df = None
        self.train = None
        self.valid = None
        self.test = None
    
    def load_data(self):
        """Load and sort data by timestamp"""
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.sort_values("timestamp")
        return self.df
    
    def time_based_split(self, train_ratio=0.7, valid_ratio=0.2):
        """
        Split data sequentially without shuffle (time-based).
        
        Args:
            train_ratio: Proportion of data for training (default: 0.7)
            valid_ratio: Proportion of data for validation (default: 0.2)
        
        Returns:
            tuple: (train, valid, test) DataFrames
        """
        if self.df is None:
            self.load_data()
        
        n = len(self.df)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        
        self.train = self.df.iloc[:train_end]
        self.valid = self.df.iloc[train_end:valid_end]
        self.test = self.df.iloc[valid_end:]
        
        return self.train, self.valid, self.test
    
    def get_features(self):
        """Return list of feature columns for model input"""
        return self.FEATURE_COLUMNS
    
    def get_splits(self):
        """
        Return train/valid/test splits. Performs split if not already done.
        
        Returns:
            tuple: (train, valid, test) DataFrames
        """
        if self.train is None:
            self.time_based_split()
        return self.train, self.valid, self.test
    
    def get_X_y(self, split="train"):
        """
        Get feature matrix X and target vector y for a specific split.
        
        Args:
            split: One of "train", "valid", or "test"
        
        Returns:
            tuple: (X, y) where X is feature DataFrame and y is target Series
        """
        if self.train is None:
            self.time_based_split()
        
        split_map = {
            "train": self.train,
            "valid": self.valid,
            "test": self.test
        }
        
        data = split_map.get(split)
        if data is None:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'valid', or 'test'")
        
        X = data[self.FEATURE_COLUMNS]
        y = data["opened"]
        return X, y
    
    def get_split_sizes(self):
        """Return dict with sizes of each split"""
        if self.train is None:
            self.time_based_split()
        
        return {
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test)
        }
    
    def get_target_distribution(self):
        """Return dict with target class distribution"""
        if self.df is None:
            self.load_data()
        
        return {
            "opened_0": int((self.df["opened"] == 0).sum()),
            "opened_1": int((self.df["opened"] == 1).sum())
        }


def evaluate_model(y_true, y_pred, split_name=""):
    """
    Calculate and print classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        split_name: Name of the split for printing (e.g., "Training", "Validation")
    
    Returns:
        dict: Dictionary with accuracy, precision, recall, and f1 scores
    """
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


# Standalone functions for backward compatibility
def load_data(path):
    """Load and sort data by timestamp (standalone function)"""
    df = pd.read_csv(path)
    df = df.sort_values("timestamp")
    return df


def get_feature_columns():
    """Return list of feature columns for model input (standalone function)"""
    return DataLoader.FEATURE_COLUMNS


def time_based_split(df, train_ratio=0.7, valid_ratio=0.2):
    """Split data sequentially without shuffle (standalone function)"""
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    
    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]
    
    return train, valid, test
