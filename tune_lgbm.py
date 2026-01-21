import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
import pandas as pd
import mlflow
from sklearn.metrics import f1_score
from models import load_data, get_feature_columns, time_based_split


DATA_PATH = "data/training_data_features_imputed.csv"
df = load_data(DATA_PATH)
train, valid, test = time_based_split(df)
features = get_feature_columns()

X_train, y_train = train[features].values, train["opened"].values
X_valid, y_valid = valid[features].values, valid["opened"].values

def objective(trial):
    """Objective function for Lightgbm tuning"""
    params = {
        "objective": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 2, 256, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0, step=0.1),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50, step=5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0, step=0.1),
        "early_stopping_rounds": 20,
    }

    model = lgb.LGBMClassifier(
        **params,
    )

    model.fit(X_train, y_train, 
        eval_set=[(X_valid, y_valid)], 
        eval_name=["valid"], 
        early_stopping_rounds=20,
        callbacks=[lgg.early_stopping(stopping_rounds=20), 
        lgb.record_evaluation(evals_result),
        LightGBMPruningCallback(trial, "binary_error")],
        verbose=-1
    )

    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)

    return f1


def run_tuning(n_trials: int = 100):
    """ Run optuna tuning for lightgbm """

    mlflow.set_experiment("lgbm_tuning")

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_hpo",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10,
         n_min_trials=3)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    with mlflow.start_run(run_name="best_lgbm_from_hpo"):

        from models_lgbm import run_lgbm_experiment
        # run _Exp already takes care of logging hyperparams and metrics
        run_lgbm_experiment(DATA_PATH, study.best_params, "best_lgbm_from_hpo")
    
    return study


if __name__ == "__main__":
    study = run_tuning(n_trials=100)

    #to show the optimization histo, ry, we can use the following
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()