import os
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "training_pipeline" / "data" / "raw" / "data.csv"
PROCESSED_DIR = PROJECT_ROOT / "training_pipeline" / "data" / "processed"
PROCESSOR_PATH = PROJECT_ROOT / "utils" / "processor.pkl"

sys.path.append(str(PROJECT_ROOT))
from utils.features import remove_cols, target_col, numerical_cols, categorical_cols
from utils.utils import PreprocessorManager, split_data_stratified


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(PROCESSOR_PATH.parent, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw data: {RAW_DATA_PATH} shape={df.shape}")

    remove_columns = [col for col in remove_cols if col in df.columns]
    if remove_columns:
        df = df.drop(columns=remove_columns)
        print(f"Dropped columns: {remove_columns}")
    else:
        print("No configured remove columns found in raw data.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the raw data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = split_data_stratified(X, y)
    print(
        f"Data split completed: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )


    # Use feature definitions from utils/features.py
    print(f"Using numerical_cols: {numerical_cols}")
    print(f"Using categorical_cols: {categorical_cols}")



    # Preprocess training and test sets
    preprocessor = PreprocessorManager(processor_path=str(PROCESSOR_PATH))
    X_train_processed = preprocessor.fit_transform_train(
        X_train,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )
    X_test_processed = preprocessor.transform_test(X_test)
    print(f"X_train processed shape: {X_train_processed.shape}")
    print(f"X_test processed shape: {X_test_processed.shape}")


    # Save outputs
    X_train_processed.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_processed.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print(f"Saved processed files to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
