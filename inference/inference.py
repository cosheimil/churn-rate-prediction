from pathlib import Path
import sys
import os
import mlflow
import mlflow.lightgbm
import pandas as pd

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UTILS_DIR = PROJECT_ROOT / "utils"
INPUT_PATH = BASE_DIR / "new_data.csv"
OUTPUT_PATH = BASE_DIR / "predictions.csv"
PROCESSOR_PATH = UTILS_DIR / "processor.pkl"
TRACKING_PATH = PROJECT_ROOT / "experiments" / "mlruns"
TRACKING_URI = "file:///" + TRACKING_PATH.as_posix().lstrip("/")
EXPERIMENT_NAME = "Training Script"
REGISTERED_MODEL_NAME = "LightGBM_BestModel"

sys.path.insert(0, str(PROJECT_ROOT))

from utils.features import categorical_cols, numerical_cols, remove_cols
from utils.utils import PreprocessorManager


def configure_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)


def load_new_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    try:
        data = pd.read_csv(INPUT_PATH)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"Input file is empty: {INPUT_PATH}. Add a header row and at least one record before running inference."
        ) from error

    columns_to_remove = [column for column in remove_cols if column in data.columns]
    if columns_to_remove:
        data = data.drop(columns=columns_to_remove)

    missing_columns = [
        column
        for column in numerical_cols + categorical_cols
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "New data is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    ordered_columns = numerical_cols + categorical_cols
    return data.loc[:, ordered_columns]


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    if not PROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Processor file not found: {PROCESSOR_PATH}")

    preprocessor = PreprocessorManager(processor_path=str(PROCESSOR_PATH))
    return preprocessor.transform_test(data)


def load_model():
    client = mlflow.tracking.MlflowClient()

    try:
        model_versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
        if model_versions:
            latest_version = max(model_versions, key=lambda version: int(version.version))
            return mlflow.lightgbm.load_model(f"models:/{REGISTERED_MODEL_NAME}/{latest_version.version}")
    except Exception:
        pass

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise FileNotFoundError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found at {TRACKING_PATH}"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        raise FileNotFoundError(
            f"No MLflow runs found for experiment '{EXPERIMENT_NAME}'"
        )

    last_error = None
    for run_id in runs["run_id"]:
        model_uri = f"runs:/{run_id}/model"
        try:
            return mlflow.lightgbm.load_model(model_uri)
        except Exception as error:
            last_error = error

    raise FileNotFoundError(
        "Unable to load a model artifact from the available MLflow runs"
    ) from last_error


def build_output_frame(raw_data: pd.DataFrame, predictions, probabilities=None) -> pd.DataFrame:
    output = raw_data.copy()
    output["prediction"] = predictions

    if probabilities is not None:
        if probabilities.ndim == 1:
            output["prediction_probability"] = probabilities
        else:
            for class_index in range(probabilities.shape[1]):
                output[f"prediction_probability_{class_index}"] = probabilities[:, class_index]

    return output


def main():
    configure_mlflow()

    raw_data = load_new_data()
    processed_data = preprocess_data(raw_data)
    model = load_model()

    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data) if hasattr(model, "predict_proba") else None

    output = build_output_frame(raw_data, predictions, probabilities)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Read input data from: {INPUT_PATH}")
    print(f"Loaded processor from: {PROCESSOR_PATH}")
    print(f"Generated predictions for {len(output)} rows")
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
