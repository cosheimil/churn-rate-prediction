import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pytest
import pandas as pd
import numpy as np

from utils.features import numerical_cols, categorical_cols, target_col, remove_cols


RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "training_pipeline" / "data" / "processed"


@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture(scope="module")
def processed_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def model():
    from lightgbm import LGBMClassifier
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    model = LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)
    return model
