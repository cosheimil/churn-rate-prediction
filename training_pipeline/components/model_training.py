from pathlib import Path
import sys
import os
import json
import joblib
import mlflow
import pandas as pd

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

tracking_path = PROJECT_ROOT / "experiments" / "mlruns"
tracking_uri = "file:///" + str(tracking_path.as_posix()).lstrip('/')
mlflow.set_tracking_uri(tracking_uri)

X_train_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_train.csv"
y_train_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "y_train.csv"
X_test_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_test.csv"
y_test_path = PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "y_test.csv"

experiment_name = "Training Script"
mlflow.set_experiment(experiment_name)

from utils.features import numerical_cols, categorical_cols
NUM_COL = numerical_cols
CAT_COL = categorical_cols

DEFAULT_PARAMS = {
    "random_state": 42,
    "verbosity": -1,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def load_best_params_from_tuning():
    try:
        tuning_exp = None
        for exp in mlflow.search_experiments():
            if exp.name == "ModelTuning":
                tuning_exp = exp
                break
        if tuning_exp is None:
            print("No ModelTuning experiment found, using default params")
            return None

        runs = mlflow.search_runs(
            experiment_ids=[tuning_exp.experiment_id],
            filter_string="tags.mlflow.runName = 'Best_Params'",
            order_by=["metrics.best_f1_macro DESC"],
        )
        if runs.empty:
            print("No Best_Params run found, using default params")
            return None

        params = {}
        for col in runs.columns:
            if col.startswith("params.best_"):
                key = col.replace("params.best_", "")
                val = runs.iloc[0][col]
                if key in ["n_estimators", "max_depth", "num_leaves"]:
                    val = int(float(val))
                elif key in ["learning_rate", "subsample", "colsample_bytree"]:
                    val = float(val)
                params[key] = val

        print(f"Loaded best params from tuning: {params}")
        return params
    except Exception as e:
        print(f"Could not load tuning params: {e}, using defaults")
        return None


if __name__ == "__main__":
    for path in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    best_params = load_best_params_from_tuning()
    final_params = {**DEFAULT_PARAMS}
    if best_params:
        final_params.update(best_params)

    RUN_NAME = "LightGBM_BestParams"

    model_params = {k: v for k, v in final_params.items() if k in [
        'random_state', 'verbosity', 'n_estimators', 'learning_rate',
        'max_depth', 'num_leaves', 'subsample', 'colsample_bytree'
    ]}
    model = LGBMClassifier(**model_params)

    metrics_dir = PROJECT_ROOT / "training_pipeline" / "metrics"
    models_dir = PROJECT_ROOT / "training_pipeline" / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "lightgbm_best_model.pkl"
    metrics_path = metrics_dir / "training_metrics.json"

    with mlflow.start_run(run_name=RUN_NAME) as run:
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        labels = sorted(y_train.unique())

        train_f1 = f1_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        train_precision = precision_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        train_recall = recall_score(y_train, y_pred_train, labels=labels, average=None, zero_division=0)
        test_precision = precision_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, labels=labels, average=None, zero_division=0)

        mlflow.log_param("model", "LightGBM")
        for k, v in final_params.items():
            mlflow.log_param(k, v)

        for label, f_val, p_val, r_val in zip(labels, train_f1, train_precision, train_recall):
            mlflow.log_metric(f"train_f1_{label}", float(f_val))
            mlflow.log_metric(f"train_precision_{label}", float(p_val))
            mlflow.log_metric(f"train_recall_{label}", float(r_val))
        for label, f_val, p_val, r_val in zip(labels, test_f1, test_precision, test_recall):
            mlflow.log_metric(f"test_f1_{label}", float(f_val))
            mlflow.log_metric(f"test_precision_{label}", float(p_val))
            mlflow.log_metric(f"test_recall_{label}", float(r_val))

        mlflow.lightgbm.log_model(model, artifact_path="model")
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model_name = "LightGBM_BestModel"
            mlflow.register_model(model_uri, registered_model_name)
            print(f"Registered model under MLflow name: {registered_model_name}")
        except Exception as register_error:
            print(f"Model registration skipped: {register_error}")

    joblib.dump(model, str(model_path))

    metrics = {
        "experiment_id": run.info.experiment_id,
        "run_id": run.info.run_id,
        "params": final_params,
    }
    for label, val in zip(labels, train_f1):
        metrics[f"train_f1_{label}"] = float(val)
    for label, val in zip(labels, train_precision):
        metrics[f"train_precision_{label}"] = float(val)
    for label, val in zip(labels, train_recall):
        metrics[f"train_recall_{label}"] = float(val)
    for label, val in zip(labels, test_f1):
        metrics[f"test_f1_{label}"] = float(val)
    for label, val in zip(labels, test_precision):
        metrics[f"test_precision_{label}"] = float(val)
    for label, val in zip(labels, test_recall):
        metrics[f"test_recall_{label}"] = float(val)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Run completed. Run name: {RUN_NAME}. Metrics saved to {metrics_path}")
    print(f"Model saved to {model_path}")
