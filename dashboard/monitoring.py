import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Churn Model Dashboard", layout="wide")
st.title("Telco Customer Churn — Model Monitoring Dashboard")

MODEL_PATH = PROJECT_ROOT / "training_pipeline" / "models" / "lightgbm_best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "training_pipeline" / "metrics" / "training_metrics.json"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "inference" / "predictions.csv"


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(str(MODEL_PATH))
    return None


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_predictions():
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    return None


model = load_model()
metrics = load_metrics()
data = load_data()
predictions = load_predictions()

tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "Data Overview", "Predictions", "Feature Importance"])

with tab1:
    st.subheader("Model Metrics")
    if metrics:
        test_metrics = {k: v for k, v in metrics.items() if k.startswith("test_")}
        if test_metrics:
            df_metrics = pd.DataFrame(
                [{"Metric": k, "Value": round(v, 4)} for k, v in test_metrics.items()]
            )
            st.dataframe(df_metrics, use_container_width=True)

            f1_yes = metrics.get("test_f1_Yes", metrics.get("test_f1_1", 0))
            f1_no = metrics.get("test_f1_No", metrics.get("test_f1_0", 0))
            col1, col2 = st.columns(2)
            col1.metric("F1 Score (Churn=Yes)", f"{f1_yes:.3f}")
            col2.metric("F1 Score (Churn=No)", f"{f1_no:.3f}")

    if model:
        st.subheader("Model Info")
        st.json({
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, "n_estimators", "N/A"),
            "boosting_type": getattr(model, "boosting_type", "N/A"),
        })

with tab2:
    st.subheader("Dataset Overview")
    if data is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(data))
        col2.metric("Features", len(data.columns) - 2)
        churn_rate = (data["Churn"] == "Yes").mean()
        col3.metric("Churn Rate", f"{churn_rate:.1%}")
        col4.metric("Avg Monthly Charges", f"${data['MonthlyCharges'].mean():.2f}")

        st.subheader("Churn Distribution")
        churn_counts = data["Churn"].value_counts()
        st.bar_chart(churn_counts)

        st.subheader("Contract Type vs Churn")
        contract_churn = pd.crosstab(data["Contract"], data["Churn"], normalize="index")
        st.dataframe(contract_churn.style.format("{:.1%}"), use_container_width=True)

        st.subheader("Tenure Distribution by Churn")
        st.scatter_chart(
            data.sample(500) if len(data) > 500 else data,
            x="tenure",
            y="MonthlyCharges",
            color="Churn",
        )

with tab3:
    st.subheader("Batch Predictions")
    if predictions is not None:
        st.dataframe(predictions, use_container_width=True)
        if "prediction" in predictions.columns:
            pred_counts = predictions["prediction"].value_counts()
            st.bar_chart(pred_counts)
    else:
        st.info("No predictions file found. Run inference first.")

    st.subheader("Single Prediction")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.5)
        total = st.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly), 10.0)
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    if st.button("Predict") and model is not None:
        from utils.features import numerical_cols, categorical_cols

        sample = {
            "SeniorCitizen": 0, "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
            "gender": "Male", "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": contract, "PaperlessBilling": "Yes", "PaymentMethod": payment,
        }
        df = pd.DataFrame([sample])

        try:
            from utils.utils import PreprocessorManager
            pm = PreprocessorManager(processor_path=str(PROJECT_ROOT / "utils" / "processor.pkl"))
            processed = pm.transform_test(df)
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0]

            st.divider()
            if isinstance(pred, str):
                churn = pred == "Yes"
            else:
                churn = bool(pred)
            if churn:
                st.error(f"Customer is likely to CHURN (probability: {prob[1]:.1%})")
            else:
                st.success(f"Customer is likely to STAY (probability: {prob[0]:.1%})")

            st.progress(float(prob[1]))
            st.caption(f"Churn probability: {prob[1]:.1%} | Retention probability: {prob[0]:.1%}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

with tab4:
    st.subheader("Feature Importance")
    if model is not None and hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        from utils.features import numerical_cols, categorical_cols

        processed_dir = PROJECT_ROOT / "training_pipeline" / "data" / "processed"
        if (processed_dir / "X_train.csv").exists():
            X_train = pd.read_csv(processed_dir / "X_train.csv")
            feature_names = X_train.columns.tolist()
        else:
            feature_names = numerical_cols + categorical_cols

        if len(importance) <= len(feature_names):
            feature_names = feature_names[:len(importance)]

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        }).sort_values("Importance", ascending=True).tail(20)

        st.bar_chart(imp_df.set_index("Feature"))

        st.subheader("Top 10 Features")
        top10 = imp_df.tail(10).iloc[::-1]
        for _, row in top10.iterrows():
            st.metric(row["Feature"], f"{row['Importance']:.4f}")
    else:
        st.info("Feature importance not available for this model.")
