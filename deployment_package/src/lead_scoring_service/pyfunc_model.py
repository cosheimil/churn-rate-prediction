from __future__ import annotations

from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd


class LeadScoringPyFuncModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow model that encapsulates preprocessing and prediction."""

    def load_context(self, context) -> None:
        self.processor = joblib.load(context.artifacts["processor"])
        self.model = joblib.load(context.artifacts["model"])
        self.feature_spec = joblib.load(context.artifacts["feature_spec"])

        self.numerical_cols = self.feature_spec["numerical_cols"]
        self.categorical_cols = self.feature_spec["categorical_cols"]
        self.remove_cols = self.feature_spec["remove_cols"]

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()

        drop_columns = [col for col in self.remove_cols if col in frame.columns]
        if drop_columns:
            frame = frame.drop(columns=drop_columns)

        required_columns = self.numerical_cols + self.categorical_cols
        missing_columns = [col for col in required_columns if col not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        frame = frame.loc[:, required_columns]

        if "scaler" in self.processor:
            scaler = self.processor["scaler"]
            numerical_columns = self.processor["numerical_columns"]
            frame[numerical_columns] = scaler.transform(frame[numerical_columns])

        if "encoders" in self.processor:
            for col, encoder in self.processor["encoders"].items():
                if encoder is None:
                    continue
                if hasattr(encoder, "classes_"):
                    frame[col] = encoder.transform(frame[col])
                else:
                    encoded = encoder.transform(frame[[col]])
                    new_cols = [f"{col}_{val}" for val in encoder.categories_[0][1:]]
                    encoded_df = pd.DataFrame(encoded, columns=new_cols, index=frame.index)
                    frame = pd.concat([frame.drop(columns=[col]), encoded_df], axis=1)

        return frame

    def predict(self, context, model_input):
        data = pd.DataFrame(model_input)
        processed = self._preprocess(data)

        predictions = self.model.predict(processed)
        output = pd.DataFrame({"prediction": predictions})

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(processed)
            if probabilities.ndim == 1:
                output["prediction_probability"] = probabilities
            else:
                for idx in range(probabilities.shape[1]):
                    output[f"prediction_probability_{idx}"] = probabilities[:, idx]

        return output
