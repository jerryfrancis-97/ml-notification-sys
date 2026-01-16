import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os 

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


# ============ Main Training Logic ============

if __name__ == "__main__":
    # Load and split data
    df = load_data("data/training_data_features_imputed.csv")
    train, valid, test = time_based_split(df)
    
    print(f"Data splits: train={len(train)}, valid={len(valid)}, test={len(test)}")
    
    # Prepare X, y
    features = get_feature_columns()
    X_train, y_train = train[features], train["opened"]
    X_valid, y_valid = valid[features], valid["opened"]
    X_test, y_test = test[features], test["opened"]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    max_iterations = 1000
    model = LogisticRegression(penalty=None, max_iter=max_iterations, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train_scaled)
    train_metrics = evaluate_model(y_train, y_train_pred, "Training")
    print("Training metrics:", train_metrics)
    # Evaluate on validation set
    y_valid_pred = model.predict(X_valid_scaled)
    valid_metrics = evaluate_model(y_valid, y_valid_pred, "Validation")
    print("Validation metrics:", valid_metrics)
    # Save feature coefficients for interpretation
    os.makedirs("experiments/logistic_regression", exist_ok=True)
    coefficients_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_[0]
    })
    coefficients_df = coefficients_df.sort_values("coefficient", key=abs, ascending=False)
    coefficients_df.to_csv(f"experiments/logistic_regression/model_coefficients_{max_iterations}.csv", index=False)
    
    print("\n" + "="*50)
    print("Feature Coefficients (sorted by importance):")
    print(coefficients_df.to_string(index=False))
    print(f"\nCoefficients saved to experiments/logistic_regression/model_coefficients_{max_iterations}.csv")
