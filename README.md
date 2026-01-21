# ML Notification System

Predict the optimal time to send push notifications to maximize user engagement while minimizing notification fatigue.

## Problem

Sending notifications at the wrong time leads to:
- Low open rates
- User fatigue and app uninstalls
- Missed engagement opportunities

This system uses ML (conditional probability estimation) to predict when each user is most likely to engage with a notification.
So, P(open | given_hour, user_features) is the best time to send notification to a given user.

We built a simulator to prepare user behavirour logs for N users based on differnt type (early bird, night owl, regular, sporadic) and added values for fatigue for number of notiifications.
Currently using data from 100 users over a span of 30 days (where 1 notification was sent per day)

Used
- MLFlow Model Registry, Tracking and Serving for experiment steup (Local Self hosting setup)
- Great expectations for data quality checks



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
python models.py          # Logistic Regression
python models_lgbm.py     # LightGBM

# 4. Hyperparameter tuning with Optuna
python tune_logreg.py    # Logistic Regression HPO
python tune_lgbm.py      # LightGBM HPO

# 5. Run decision engine
python decision_engine.py
```

## Experiment Results

### Model Comparison (Validation Set)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.644 | 0.354 | 0.497 | 0.413 |
| LightGBM (default) | 0.686 | 0.412 | 0.576 | **0.481** |
| LightGBM (HPO best) | **0.765** | **0.778** | 0.093 | 0.166 |

### Key Observations

- **LightGBM (default)** achieves the best F1 score (0.481) with balanced precision/recall
- **LightGBM (HPO)** optimized for F1 but converged to high-precision/low-recall (conservative predictions)
- **Logistic Regression** provides a solid baseline with interpretable coefficients

### Features Used

| Feature | Description |
|---------|-------------|
| `hour`, `hour_sin`, `hour_cos` | Time of day (cyclical encoding) |
| `day_of_week`, `is_weekend` | Day information |
| `num_notifications_last_24h` | Notification fatigue indicator |
| `delay_since_last_open_notification` | Recency of engagement |
| `user_open_rate` | Historical user engagement rate |
| `user_hour_open_rate` | User's hour-specific engagement pattern |

### Data Split

- **Train**: 2,100 samples (70%)
- **Validation**: 599 samples (20%)
- **Test**: 301 samples (10%)
