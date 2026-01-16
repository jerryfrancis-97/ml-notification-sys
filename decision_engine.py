

import numpy as np
from datetime import datetime
import joblib


# ============ Mock Feature Store ============

FEATURE_STORE = {
    "U_1": {
        "user_open_rate": 0.45,
        "user_hour_open_rate": 0.38,
        "num_notifications_last_24h": 1,
        "delay_since_last_open_notification": 12,
        "notifications_sent_today": 1
    },
    "U_2": {
        "user_open_rate": 0.62,
        "user_hour_open_rate": 0.55,
        "num_notifications_last_24h": 0,
        "delay_since_last_open_notification": 48,
        "notifications_sent_today": 0
    },
    "U_3": {
        "user_open_rate": 0.28,
        "user_hour_open_rate": 0.22,
        "num_notifications_last_24h": 2,
        "delay_since_last_open_notification": 6,
        "notifications_sent_today": 2
    },
    "U_4": {
        "user_open_rate": 0.71,
        "user_hour_open_rate": 0.65,
        "num_notifications_last_24h": 1,
        "delay_since_last_open_notification": 24,
        "notifications_sent_today": 1
    }
}


def get_user_features(user_id, feature_store):
    """
    Retrieve user features from the feature store
    """
    return feature_store.get(user_id)


def build_feature_vector(hour, user_features):
    """
    Build a complete featurre vector for a given hour and usr features.
    """
    # Get current day info
    now = datetime.now()
    day_of_week = now.weekday()  # 0=Monday, 6=Sunday
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Compute cyclical hour encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Build feature vector in the same order as training
    # ["hour", "day_of_week", "is_weekend",
    #  "num_notifications_last_24h", "delay_since_last_open_notification",
    #  "user_open_rate", "user_hour_open_rate", "hour_sin", "hour_cos"]
    features = np.array([
        hour,
        day_of_week,
        is_weekend,
        user_features["num_notifications_last_24h"],
        user_features["delay_since_last_open_notification"],
        user_features["user_open_rate"],
        user_features["user_hour_open_rate"],
        hour_sin,
        hour_cos
    ])
    
    return features



def select_best_hour(model, scaler, user_id, feature_store, candidate_hours=None):
    """
    Select the best hour to send notification for a user.
    """
    if candidate_hours is None:
        candidate_hours = list(range(7, 23))  # 7 AM to 10 PM
    
    # Get user features
    user_features = get_user_features(user_id, feature_store)
    if user_features is None:
        return {
            "best_hour": None,
            "probability": 0.0,
            "all_predictions": {},
            "error": f"User {user_id} not found in feature store"
        }
    
    # Build feature vectors for all candidate hours
    feature_vectors = []
    for hour in candidate_hours:
        fv = build_feature_vector(hour, user_features)
        feature_vectors.append(fv)
    
    # Stack and scale features
    X = np.array(feature_vectors)
    X_scaled = scaler.transform(X)
    
    # Get probabilities for class 1 (opened)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Find best hour using argmax
    best_idx = np.argmax(probabilities)
    best_hour = candidate_hours[best_idx]
    best_probability = probabilities[best_idx]
    
    # Create predictions dict for all hours
    all_predictions = {
        hour: float(prob) 
        for hour, prob in zip(candidate_hours, probabilities)
    }
    
    return {
        "best_hour": best_hour,
        "probability": float(best_probability),
        "all_predictions": all_predictions
    }



def decide_notification(model, scaler, user_id, feature_store, 
                        candidate_hours=None, min_probability=0.15, max_notifications_per_day=2):
    """
    Main decision function that combines hour selection with guardrails.
    
    Args:
        model: Trained sklearn model
        scaler: Fitted StandardScaler
        user_id: User identifier
        feature_store: Dictionary containing user features
        candidate_hours: List of hours to consider (default: 7-22)
        min_probability: Minimum required probability (default: 0.15)
        max_notifications_per_day: Max notifications per day (default: 2)
    
    Returns:
        dict with complete decision information
    """
    # Step 1: Select best hour
    selection = select_best_hour(model, scaler, user_id, feature_store, candidate_hours)
    
    if "error" in selection:
        return {
            "user_id": user_id,
            "send": False,
            "hour": None,
            "probability": 0.0,
            "reason": "USER_NOT_FOUND",
            "details": selection["error"],
            "all_predictions": {}
        }
    
    best_hour = selection["best_hour"]
    probability = selection["probability"]

    
    return {
        "user_id": user_id,
        "send": guardrail_result["should_send"],
        "hour": best_hour,
        "probability": probability,
        "reason": guardrail_result["reason"],
        "details": guardrail_result["details"],
        "all_predictions": selection["all_predictions"]
    }


