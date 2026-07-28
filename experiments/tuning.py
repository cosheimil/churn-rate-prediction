# tune lightgbm with optuna

import optuna
import mlflow
import pandas as pd
import os
from lightgbm import LGBMClassifier

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
import sys
import os

# Set base dir using this script location (works from any current working directory)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Set tracking URI to your experiments folder (use local path on Windows)
tracking_path = PROJECT_ROOT / "experiments" / "mlruns"
tracking_uri = "file:///" + str(tracking_path.as_posix()).lstrip('/')
mlflow.set_tracking_uri(tracking_uri)

# Define experiment name
experiment_name = "ModelTuning"
mlflow.set_experiment(experiment_name)

# Paths to processed train data (same as model_training.py)
X_train_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_train.csv"
y_train_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "y_train.csv"

# Paths to processed test data
X_test_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_test.csv"
y_test_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "y_test.csv"






def objective(trial):
    if not X_train_path.exists() or not y_train_path.exists() or not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Data files not found. Expected:\n  {X_train_path}\n  {y_train_path}\n  {X_test_path}\n  {y_test_path}\n" +
            "Run from the project root or adjust paths accordingly."
        )

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    num_leaves = trial.suggest_int('num_leaves', 20, 100)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

    with mlflow.start_run(run_name=f"Trial_{trial.number}"):
        model = LGBMClassifier(
            random_state=42,
            verbosity=-1,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )
        model.fit(X_train, y_train)

        # Training predictions
        y_pred_train = model.predict(X_train)
        labels = sorted(y_train.unique())

        # Test predictions
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        precision_train = precision_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        precision_test = precision_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)

        # Log parameters
        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("num_leaves", num_leaves)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)

        # Log training metrics
        for label, p, r in zip(labels, precision_train, recall_train):
            mlflow.log_metric(f"train_precision_{label}", float(p))
            mlflow.log_metric(f"train_recall_{label}", float(r))

        # Log test metrics
        for label, p, r in zip(labels, precision_test, recall_test):
            mlflow.log_metric(f"test_precision_{label}", float(p))
            mlflow.log_metric(f"test_recall_{label}", float(r))

        # Calculate macro F1 for optimization
        f1_test = f1_score(y_test, y_pred_test, average='macro')
        mlflow.log_metric("test_f1_macro", float(f1_test))

        print(f"Trial {trial.number} completed. Test F1 Macro: {f1_test}")

        return f1_test

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Log best parameters
    with mlflow.start_run(run_name="Best_Params"):
        best_params = study.best_params
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", value)
        mlflow.log_metric("best_f1_macro", study.best_value)
        print(f"Best parameters: {best_params}")
        print(f"Best F1 Macro: {study.best_value}")