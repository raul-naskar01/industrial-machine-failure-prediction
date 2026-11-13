"""Streamlit interface for the trained machine failure prediction model."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_machine_failure_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Machine Failure Predictor", page_icon="⚙️", layout="wide")
st.title("⚙️ Industrial Machine Failure Prediction")
st.caption("Predictive maintenance using operating-condition and sensor measurements.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Trained model files are missing. Train the model first.")
    st.code("python train.py", language="bash")
    st.stop()

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata

model, metadata = load_artifacts()
metrics = metadata.get("metrics", {})

with st.sidebar:
    st.header("Model information")
    st.write(f"**Selected model:** {metadata.get('best_model', 'Unknown')}")
    st.write(f"**Training rows:** {metadata.get('training_rows', '—'):,}")
    st.write(f"**Testing rows:** {metadata.get('testing_rows', '—'):,}")
    if metrics:
        st.metric("Test ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        st.metric("Test F1-score", f"{metrics.get('f1', 0):.3f}")

st.subheader("Enter machine operating conditions")
feature_names = metadata["features"]
defaults = {
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 120,
}
values = {}
columns = st.columns(2)
for index, feature in enumerate(feature_names):
    with columns[index % 2]:
        default = defaults.get(feature)
        if feature in metadata.get("categorical_features", []):
            values[feature] = st.selectbox(feature, options=["L", "M", "H"], index=["L", "M", "H"].index(default or "M"))
        else:
            if default is None:
                default = 0.0
            values[feature] = st.number_input(feature, value=float(default))

if st.button("Predict machine failure", type="primary", use_container_width=True):
    input_df = pd.DataFrame([values], columns=feature_names)
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    st.divider()
    left, right = st.columns(2)
    with left:
        if prediction == 1:
            st.error("⚠️ Potential machine failure detected")
            st.write("Consider inspecting the machine and reviewing its maintenance schedule.")
        else:
            st.success("✅ No machine failure predicted")
            st.write("The model did not identify a failure for the supplied measurements.")
    with right:
        st.metric("Estimated failure probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("View submitted sensor values"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.subheader("Model comparison")
comparison = pd.DataFrame(metadata.get("all_model_results", []))
if not comparison.empty:
    st.dataframe(comparison.set_index("model").style.format(precision=3), use_container_width=True)
else:
    st.info("Model comparison will appear after training.")

st.caption("This is an academic decision-support demonstration, not a certified industrial safety system.")
