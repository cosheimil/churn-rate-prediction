from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "deployment_package" / "src"
sys.path.insert(0, str(SRC_PATH))

from lead_scoring_service.pyfunc_model import LeadScoringPyFuncModel  # type: ignore[reportMissingImports]


TRACKING_PATH = Path(os.getenv("TRACKING_PATH", PROJECT_ROOT / "experiments" / "mlruns"))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///" + TRACKING_PATH.as_posix().lstrip("/"))
MODEL_FILE_PATH = Path(os.getenv("MODEL_FILE_PATH", PROJECT_ROOT / "training_pipeline" / "models" / "lightgbm_best_model.pkl"))
PROCESSOR_FILE_PATH = Path(os.getenv("PROCESSOR_FILE_PATH", PROJECT_ROOT / "utils" / "processor.pkl"))
FEATURES_MODULE_PATH = PROJECT_ROOT / "utils" / "features.py"
RAW_SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "LeadScoringService")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "DeploymentPackaging")
TEMP_DIR = PROJECT_ROOT / "deployment_package" / "artifacts"
FEATURE_SPEC_PATH = TEMP_DIR / "feature_spec.pkl"


def load_feature_spec() -> dict[str, list[str]]:
    namespace: dict[str, object] = {}
    code = FEATURES_MODULE_PATH.read_text(encoding="utf-8")
    exec(code, namespace)

    numerical_cols = namespace["numerical_cols"]
    categorical_cols = namespace["categorical_cols"]
    remove_cols = namespace["remove_cols"]

    return {
        "numerical_cols": list(numerical_cols),
        "categorical_cols": list(categorical_cols),
        "remove_cols": list(remove_cols),
    }


def create_input_example(feature_spec: dict[str, list[str]]) -> pd.DataFrame:
    required_cols = feature_spec["remove_cols"] + feature_spec["numerical_cols"] + feature_spec["categorical_cols"]

    if RAW_SAMPLE_PATH.exists():
        source = pd.read_csv(RAW_SAMPLE_PATH)
        available = [col for col in required_cols if col in source.columns]
        if len(available) == len(required_cols):
            return source.loc[:, required_cols].head(3)

    # Fallback empty frame with schema-only columns if sample raw data is missing
    return pd.DataFrame(columns=required_cols)


def validate_paths() -> None:
    missing = [path for path in [MODEL_FILE_PATH, PROCESSOR_FILE_PATH, FEATURES_MODULE_PATH] if not path.exists()]
    if missing:
        lines = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required files not found:\n{lines}")


def main() -> None:
    validate_paths()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    feature_spec = load_feature_spec()
    joblib.dump(feature_spec, FEATURE_SPEC_PATH)

    input_example = create_input_example(feature_spec)
    output_example = pd.DataFrame({"prediction": [0]}) if input_example.empty else pd.DataFrame({"prediction": [0] * len(input_example)})
    signature = infer_signature(input_example, output_example)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="register_lead_scoring_service") as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LeadScoringPyFuncModel(),
            artifacts={
                "model": str(MODEL_FILE_PATH),
                "processor": str(PROCESSOR_FILE_PATH),
                "feature_spec": str(FEATURE_SPEC_PATH),
            },
            code_paths=[str(PROJECT_ROOT / "deployment_package" / "src")],
            pip_requirements=[
                "pandas==3.0.2",
                "numpy==2.4.4",
                "scikit-learn==1.8.0",
                "lightgbm==4.6.0",
                "mlflow==3.10.1",
                "joblib>=1.3.0",
                "cloudpickle>=3.0.0",
            ],
            input_example=input_example,
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Registered model name: {REGISTERED_MODEL_NAME}")
        print("Model package logged with artifacts, requirements, and signature.")


if __name__ == "__main__":
    main()
