from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.utils import PreprocessorManager
from utils.features import numerical_cols, categorical_cols, remove_cols


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1)


MODEL_PATH = PROJECT_ROOT / "training_pipeline" / "models" / "lightgbm_best_model.pkl"
PROCESSOR_PATH = PROJECT_ROOT / "utils" / "processor.pkl"

_model = None
_preprocessor = None


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(str(MODEL_PATH))


def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    global _preprocessor
    if _preprocessor is None:
        if not PROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Processor not found at {PROCESSOR_PATH}")
        _preprocessor = PreprocessorManager(processor_path=str(PROCESSOR_PATH))

    frame = data.copy()
    drop_columns = [c for c in remove_cols if c in frame.columns]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)

    float_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    for col in float_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    required = numerical_cols + categorical_cols
    available = [c for c in required if c in frame.columns]
    frame = frame.loc[:, available]

    return _preprocessor.transform_test(frame)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    yield
    _model = None


app = FastAPI(title="Telco Churn Prediction API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(_model is not None)}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        frame = pd.DataFrame(request.rows)
        processed = preprocess_input(frame)
        preds = _model.predict(processed)
        probs = _model.predict_proba(processed) if hasattr(_model, "predict_proba") else None

        results = []
        for i in range(len(preds)):
            result = {
                "prediction": int(preds[i]) if hasattr(preds[i], "item") else preds[i],
                "churn": bool(preds[i]),
            }
            if probs is not None:
                result["probability"] = float(probs[i][1]) if probs.shape[1] > 1 else float(probs[i])
            results.append(result)

        return {"predictions": results}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "model_type": type(_model).__name__,
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
    }


@app.post("/predict/explain")
def predict_explain(request: PredictRequest) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        import shap

        frame = pd.DataFrame(request.rows)
        processed = preprocess_input(frame)

        preds = _model.predict(processed)
        probs = _model.predict_proba(processed) if hasattr(_model, "predict_proba") else None

        explainer = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(processed)

        results = []
        for i in range(len(preds)):
            result = {
                "prediction": int(preds[i]) if hasattr(preds[i], "item") else preds[i],
                "churn": bool(preds[i]),
            }
            if probs is not None:
                result["probability"] = float(probs[i][1]) if probs.shape[1] > 1 else float(probs[i])

            if isinstance(shap_values, list):
                sv = shap_values[1][i] if len(shap_values) > 1 else shap_values[0][i]
            else:
                sv = shap_values[i]

            feature_contributions = [
                {"feature": str(f), "contribution": float(v)}
                for f, v in zip(processed.columns, sv)
            ]
            feature_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            result["shap_values"] = feature_contributions[:10]
            results.append(result)

        return {"predictions": results}
    except ImportError:
        raise HTTPException(status_code=500, detail="SHAP is not installed. Install with: pip install shap")
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/stats")
def stats() -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return {
        "model_type": type(_model).__name__,
        "features": {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
        },
        "total_features": len(numerical_cols) + len(categorical_cols),
    }
