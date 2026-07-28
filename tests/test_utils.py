import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile
import os
import joblib

from utils.utils import (
    PreprocessorManager,
    split_data_stratified,
    check_multicollinearity,
    target_balance,
)


class TestPreprocessorManager:
    def test_fit_transform_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor_path = os.path.join(tmpdir, "processor.pkl")
            pm = PreprocessorManager(processor_path=processor_path)

            X_train = pd.DataFrame({
                "age": [25, 30, 35, 40, 45],
                "gender": ["M", "F", "M", "F", "M"],
                "city": ["NY", "LA", "NY", "SF", "LA"],
            })
            num_cols = ["age"]
            cat_cols = ["gender", "city"]

            X_processed = pm.fit_transform_train(
                X_train,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
            )

            assert os.path.exists(processor_path)
            assert X_processed.shape[0] == 5
            assert "age" in X_processed.columns
            assert isinstance(X_processed["age"].iloc[0], float)

    def test_transform_test_reproduces_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor_path = os.path.join(tmpdir, "processor.pkl")
            pm = PreprocessorManager(processor_path=processor_path)

            X_train = pd.DataFrame({
                "age": [25, 30, 35, 40, 45, 50],
                "gender": ["M", "F", "M", "F", "M", "F"],
                "city": ["NY", "LA", "NY", "SF", "LA", "NY"],
            })

            pm.fit_transform_train(X_train, numerical_cols=["age"], categorical_cols=["gender", "city"])

            X_test = pd.DataFrame({
                "age": [28, 33],
                "gender": ["F", "M"],
                "city": ["LA", "NY"],
            })

            pm2 = PreprocessorManager(processor_path=processor_path)
            X_test_processed = pm2.transform_test(X_test)

            assert X_test_processed.shape[0] == 2

    def test_missing_processor_raises(self):
        pm = PreprocessorManager(processor_path="/nonexistent/processor.pkl")
        X_test = pd.DataFrame({"age": [25]})
        with pytest.raises(FileNotFoundError):
            pm.transform_test(X_test)

    def test_binary_column_passthrough(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor_path = os.path.join(tmpdir, "processor.pkl")
            pm = PreprocessorManager(processor_path=processor_path)

            X_train = pd.DataFrame({
                "val": [0, 1, 0, 1, 0],
                "cat": ["A", "B", "A", "B", "A"],
            })

            X_processed = pm.fit_transform_train(X_train, categorical_cols=["val", "cat"])

            assert 0 in X_processed["val"].values or 1 in X_processed["val"].values


class TestSplitData:
    def test_stratified_split_preserves_ratio(self):
        X = pd.DataFrame({"feat": range(100)})
        y = pd.Series([0] * 80 + [1] * 20)

        X_train, X_test, y_train, y_test = split_data_stratified(X, y, test_size=0.2, random_state=42)

        train_ratio = (y_train == 1).mean()
        test_ratio = (y_test == 1).mean()

        assert abs(train_ratio - 0.2) < 0.05
        assert abs(test_ratio - 0.2) < 0.05
        assert len(X_test) == 20
        assert len(X_train) == 80


class TestVIF:
    def test_multicollinearity_check(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
        })
        vif = check_multicollinearity(df)
        assert len(vif) == 2
        assert "VIF" in vif.columns


class TestTargetBalance:
    def test_target_balance_output(self):
        df = pd.DataFrame({"target": [0, 0, 0, 1, 1]})
        balance = target_balance(df, "target")
        assert balance.iloc[0]["count"] == 3
        assert balance.iloc[1]["count"] == 2
        assert abs(balance.iloc[0]["pct"] - 0.6) < 0.01
