from pathlib import Path
import sys
import os
import mlflow
import pandas as pd

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

tracking_path = PROJECT_ROOT / "experiments" / "mlruns"
tracking_uri = "file:///" + str(tracking_path.as_posix()).lstrip('/')
mlflow.set_tracking_uri(tracking_uri)

X_train_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_train.csv"
y_train_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "y_train.csv"
X_test_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_test.csv"
y_test_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "y_test.csv"

experiment_name = "BaselineModels"
mlflow.set_experiment(experiment_name)


def load_data():
    for path in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")
    return X_train, y_train, X_test, y_test


def create_model(model_name):
    if model_name == "LightGBM":
        return LGBMClassifier(random_state=42, verbosity=-1, n_estimators=100, learning_rate=0.1)
    elif model_name == "XGBoost":
        return XGBClassifier(random_state=42, verbosity=0, n_estimators=100, learning_rate=0.3, max_depth=6)
    elif model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_estimators=100)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def log_params(model_name):
    params = {"model": model_name}
    if model_name == "LightGBM":
        params.update({"n_estimators": 100, "learning_rate": 0.1, "random_state": 42})
    elif model_name == "XGBoost":
        params.update({"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6, "random_state": 42})
    elif model_name == "LogisticRegression":
        params.update({"max_iter": 1000, "random_state": 42})
    elif model_name == "RandomForest":
        params.update({"n_estimators": 100, "random_state": 42})
    for k, v in params.items():
        mlflow.log_param(k, v)


def train_and_log(model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        model = create_model(model_name)
        model.fit(X_train, y_train)

        labels = sorted(y_train.unique())
        log_params(model_name)

        for dataset_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            precision = precision_score(y, y_pred, labels=labels, average=None, zero_division=0)
            recall = recall_score(y, y_pred, labels=labels, average=None, zero_division=0)
            f1 = f1_score(y, y_pred, labels=labels, average=None, zero_division=0)
            for label, p, r, f in zip(labels, precision, recall, f1):
                mlflow.log_metric(f"{dataset_name}_precision_{label}", float(p))
                mlflow.log_metric(f"{dataset_name}_recall_{label}", float(r))
                mlflow.log_metric(f"{dataset_name}_f1_{label}", float(f))

        print(f"Run completed: {model_name}")

    return model


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    models_to_train = ["LightGBM", "XGBoost", "LogisticRegression", "RandomForest"]
    for name in models_to_train:
        train_and_log(name, X_train, y_train, X_test, y_test)

    print(f"\nAll 4 baseline models trained and logged to MLflow experiment: {experiment_name}")
