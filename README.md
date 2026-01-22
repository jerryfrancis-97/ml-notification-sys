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


## Data Versioning with DVC

We use [DVC (Data Version Control)](https://dvc.org/) to track and version large data files alongside Git.

### Initial Setup (one-time)

```bash
# Install DVC
pip install dvc

# Initialize DVC in your repo (already done)
dvc init

# Add remote storage (local folder in this case)
dvc remote add -d myremote data/files
```

### Adding a New Dataset Version

Whenever you update or add new data files:

```bash
# 1. Add the data file to DVC tracking
dvc add data/your_new_file.csv

# 2. Stage the .dvc pointer file and updated .gitignore
git add data/your_new_file.csv.dvc data/.gitignore

# 3. Commit the changes to Git
git commit -m "Add new dataset version: your_new_file.csv"

# 4. Push data to DVC remote storage
dvc push

# 5. Push code changes to Git remote
git push
```

### Pulling Data (for collaborators)

```bash
# After cloning the repo, pull the actual data files
git pull
dvc pull
```

### Switching Between Dataset Versions

```bash
# Checkout a specific Git commit/tag
git checkout <commit-hash>

# Pull the corresponding data version
dvc checkout
```