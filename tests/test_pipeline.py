import pytest
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDataIngestion:
    def test_ingestion_produces_output(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
            output_path = Path(tmpdir) / "output.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "training_pipeline" / "components" / "data_ingestion.py"),
                    "--input_path", str(input_path),
                    "--output_path", str(output_path),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert output_path.exists()
            import pandas as pd
            df = pd.read_csv(output_path)
            assert len(df) > 0
            assert "customerID" in df.columns
            assert "Churn" in df.columns


class TestPreprocessing:
    def test_preprocessing_produces_all_files(self):
        processed_dir = PROJECT_ROOT / "training_pipeline" / "data" / "processed"
        assert (processed_dir / "X_train.csv").exists()
        assert (processed_dir / "X_test.csv").exists()
        assert (processed_dir / "y_train.csv").exists()
        assert (processed_dir / "y_test.csv").exists()

    def test_processor_saved(self):
        processor_path = PROJECT_ROOT / "utils" / "processor.pkl"
        assert processor_path.exists()

    def test_train_test_no_overlap(self):
        import pandas as pd
        X_train = pd.read_csv(PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_train.csv")
        X_test = pd.read_csv(PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_test.csv")
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[0] + X_test.shape[0] == 7043


class TestModel:
    def test_model_saved(self):
        model_path = PROJECT_ROOT / "training_pipeline" / "models" / "lightgbm_best_model.pkl"
        assert model_path.exists()

    def test_metrics_saved(self):
        metrics_path = PROJECT_ROOT / "training_pipeline" / "metrics" / "training_metrics.json"
        assert metrics_path.exists()
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "params" in metrics
        assert any(k.startswith("test_f1_") for k in metrics.keys())

    def test_model_predicts(self):
        import joblib
        import pandas as pd
        model = joblib.load(str(PROJECT_ROOT / "training_pipeline" / "models" / "lightgbm_best_model.pkl"))
        X_test = pd.read_csv(PROJECT_ROOT / "training_pipeline" / "data" / "processed" / "X_test.csv")
        preds = model.predict(X_test.head(5))
        assert len(preds) == 5
