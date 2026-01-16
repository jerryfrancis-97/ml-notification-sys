import numpy as np
from datetime import datetime
import joblib
import mlflow
from mlflow.tracking import MlflowClient


def load_model_from_mlflow(experiment_name, run_id=None, model_name=None):
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Get run - either specific run_id or latest
    if run_id is None:
        # Get the latest run from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        print(f"Runs found in experiment '{experiment_name}':")
        for run in runs:
            print(run.info.run_id)

        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        run = runs[0]
        run_id = run.info.run_id
    else:
        run = client.get_run(run_id)
    
    # Load the model
    model_uri = f"runs:/{run_id}/model"
    
    # Try to load as sklearn model
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback: search for model artifacts
        artifacts = client.list_artifacts(run_id)
        model_artifacts = [a for a in artifacts if 'model' in a.path.lower()]
        if model_artifacts:
            model_uri = f"runs:/{run_id}/{model_artifacts[0].path}"
            model = mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError(f"Could not find model in run {run_id}")
    
    print(f"Loaded model from experiment: {experiment_name}")
    print(f"Run ID: {run_id}")
    print(f"Run name: {run.info.run_name}")
    
    return {
        "model": model,
        "run_id": run_id,
        "run_name": run.info.run_name,
        "experiment_name": experiment_name
    }


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



def apply_guardrails(user_id, probability, feature_store, 
                     min_probability=0.15, max_notifications_per_day=2):
    """
    Apply guardrails to determine if notification should be sent.
    """
    user_features = get_user_features(user_id, feature_store)
    
    if user_features is None:
        return {"should_send": False, "reason": "USER_NOT_FOUND", "details": "User not found"}
    
    # Check 1: Minimum probability threshold
    if probability < min_probability:
        return {
            "should_send": False,
            "reason": "LOW_PROBABILITY",
            "details": f"Probability {probability:.3f} < threshold {min_probability}"
        }
    
    # Check 2: Maximum notifications per day
    notifications_sent_today = user_features.get("notifications_sent_today", 0)
    if notifications_sent_today >= max_notifications_per_day:
        return {
            "should_send": False,
            "reason": "MAX_NOTIFICATIONS_REACHED",
            "details": f"Already sent {notifications_sent_today} notifications today"
        }
    
    return {
        "should_send": True,
        "reason": "OK",
        "details": f"Probability: {probability:.3f}, Notifications today: {notifications_sent_today}"
    }


def decide_notification(model, scaler, user_id, feature_store, 
                        candidate_hours=None, min_probability=0.15, max_notifications_per_day=2):
    """
    Main decision function that combines hour selection with guardrails.
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
    
    # Step 2: Apply guardrails
    guardrail_result = apply_guardrails(
        user_id, probability, feature_store,
        min_probability=min_probability,
        max_notifications_per_day=max_notifications_per_day
    )
    
    return {
        "user_id": user_id,
        "send": guardrail_result["should_send"],
        "hour": best_hour,
        "probability": probability,
        "reason": guardrail_result["reason"],
        "details": guardrail_result["details"],
        "all_predictions": selection["all_predictions"]
    }



if __name__ == "__main__":

    from models import load_data, get_feature_columns, time_based_split
    
    print("="*60)
    print("Notification Decision Engine")
    print("="*60)
    
    # Load model from MLflow
    print("\nLoading model from MLflow...")
    try:
        result = load_model_from_mlflow("notification_logreg")
        model = result["model"]
        print(f"Model loaded successfully!")
    except Exception as e:
        raise Exception(f"Could not load from MLflow: {e}")

    

    print("Running decisions for all users:")
    print("="*60)
    
    for user_id in FEATURE_STORE.keys():
        decision = decide_notification(model, scaler, user_id, FEATURE_STORE)
        
        print(f"\nUser: {user_id}")
        print(f"  Send: {decision['send']}")
        print(f"  Best Hour: {decision['hour']}")
        print(f"  Probability: {decision['probability']:.3f}")
        print(f"  Reason: {decision['reason']}")
        print(f"  Details: {decision['details']}")

