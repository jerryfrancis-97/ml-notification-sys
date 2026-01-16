# ML Notification System

Predict the optimal time to send push notifications to maximize user engagement while minimizing notification fatigue.

## Problem

Sending notifications at the wrong time leads to:
- Low open rates
- User fatigue and app uninstalls
- Missed engagement opportunities

This system uses ML (conditional probability estimation) to predict when each user is most likely to engage with a notification.
So, P(open | given_hour, user_features) is the best time to send notification to a given user.


## Setup

```bash
# Create virtual environment
python -m venv ml_notifi_env
source ml_notifi_env/bin/activate  # Linux/Mac
# or: ml_notifi_env\Scripts\activate  # Windows

# Install dependencies
pip install pandas numpy scikit-learn mlflow matplotlib seaborn great_expectations python-dotenv

# Start MLflow server (optional, for experiment tracking)
mlflow server --host 0.0.0.0 --port 5000
```

## Quick Start

```bash
# 1. Generate synthetic data
python simulator.py
python generate_logs.py

# 2. Feature engineering + imputation
python feature_engg.py
python impute_features.py

# 3. Train model (logs to MLflow)
python models.py

# 4. Run decision engine
python decision_engine.py
```

