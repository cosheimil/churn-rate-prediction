from utils.features import numerical_cols, categorical_cols, target_col, remove_cols


class TestFeatures:
    def test_numerical_cols_not_empty(self):
        assert len(numerical_cols) == 4
        assert isinstance(numerical_cols, list)

    def test_categorical_cols_not_empty(self):
        assert len(categorical_cols) == 15
        assert isinstance(categorical_cols, list)

    def test_target_col_defined(self):
        assert target_col == "Churn"

    def test_remove_cols_contains_customerid(self):
        assert "customerID" in remove_cols
        assert len(remove_cols) == 1

    def test_no_overlap_numerical_categorical(self):
        assert set(numerical_cols).isdisjoint(set(categorical_cols))

    def test_all_features_in_raw_data(self):
        import pandas as pd
        from pathlib import Path
        df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "raw" / "telco_churn.csv")
        for col in numerical_cols + categorical_cols + [target_col] + remove_cols:
            assert col in df.columns, f"Column {col} not found in raw data"
