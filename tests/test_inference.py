import pytest
import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPreprocessorReuse:
    def test_preprocessor_on_new_data(self):
        from utils.utils import PreprocessorManager
        import pandas as pd

        processor_path = PROJECT_ROOT / "utils" / "processor.pkl"
        pm = PreprocessorManager(processor_path=str(processor_path))

        new_data = pd.DataFrame({
            "SeniorCitizen": [0, 1],
            "tenure": [12, 48],
            "MonthlyCharges": [70.0, 100.0],
            "TotalCharges": [840.0, 4800.0],
            "gender": ["Male", "Female"],
            "Partner": ["No", "Yes"],
            "Dependents": ["No", "No"],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
        })

        processed = pm.transform_test(new_data)
        assert processed.shape[0] == 2
        assert processed.shape[1] > 0


class TestInference:
    def test_inference_script_runs(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "inference" / "inference.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Inference failed: {result.stderr}"
        assert "predictions" in result.stdout.lower()

    def test_predictions_output_exists(self):
        output_path = PROJECT_ROOT / "inference" / "predictions.csv"
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) > 0
        assert "prediction" in df.columns
