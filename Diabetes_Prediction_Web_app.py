# -*- coding: utf-8 -*-
"""
Diabetes Prediction Web App (Streamlit)

Run:  streamlit run Diabetes_Prediction_Web_app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Model details shown in the sidebar (from train.ipynb).
# If you retrain the model, update these numbers.
# ----------------------------------------------------------------------------
MODEL_NAME = "Support Vector Machine (SVC, linear kernel)"
TEST_ACCURACY = 92.9      # %
TRAIN_ACCURACY = 92.3     # %
DATASET_SIZE = 800

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.sav")

# Feature names (same order as training data)
feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #0f766e 0%, #2563eb 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.1rem; }
    .hero p  { color: #e0f2fe; margin: 6px 0 0 0; font-size: 1.05rem; }

    .result-card {
        padding: 22px 26px;
        border-radius: 14px;
        margin-top: 12px;
        font-size: 1.35rem;
        font-weight: 600;
    }
    .result-ok  { background: #dcfce7; color: #166534; border-left: 8px solid #16a34a; }
    .result-bad { background: #fee2e2; color: #991b1b; border-left: 8px solid #dc2626; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Load model (cached so it loads only once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Prediction function
# ----------------------------------------------------------------------------
def diabetes_prediction(model, input_data):
    """Returns (is_diabetic, risk_score between 0 and 1)."""
    input_df = pd.DataFrame([input_data], columns=feature_names)

    prediction = model.predict(input_df)[0]

    # The saved SVM has no predict_proba, so we turn its decision score
    # (distance from the decision boundary) into a 0-1 score with a sigmoid.
    # 0.5 is exactly the boundary, so score > 0.5 matches "diabetic".
    decision = float(model.decision_function(input_df)[0])
    risk_score = 1 / (1 + np.exp(-decision))

    return int(prediction) == 1, risk_score


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("📊 Model Information")

    st.sidebar.metric("Test Accuracy", f"{TEST_ACCURACY:.1f}%")
    st.sidebar.metric("Training Accuracy", f"{TRAIN_ACCURACY:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model used**  \n{MODEL_NAME}")
    st.sidebar.markdown(f"**Dataset**  \n{DATASET_SIZE} patient records")
    st.sidebar.markdown("**Inputs**  \n" + ", ".join(feature_names))

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Enter the patient details in the form and click **Predict**."
    )
    st.sidebar.warning(
        "⚠️ For educational purposes only. This is not a medical diagnosis."
    )


# ----------------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------------
def main():
    render_sidebar()

    st.markdown(
        """
        <div class="hero">
            <h1>🩺 Diabetes Prediction Web App</h1>
            <p>Enter the patient's details to check whether the person is likely to be diabetic.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Make sure `model/trained_model.sav` exists.")
        st.stop()

    model = load_model()

    st.subheader("Patient Details")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "🤰 Number of Pregnancies", min_value=0, max_value=20, value=1, step=1
            )
            glucose = st.number_input(
                "🍬 Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0
            )
            blood_pressure = st.number_input(
                "❤️ Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0
            )
            skin_thickness = st.number_input(
                "📏 Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=25.0, step=1.0
            )

        with col2:
            insulin = st.number_input(
                "💉 Insulin Level (mu U/ml)", min_value=0.0, max_value=900.0, value=100.0, step=1.0
            )
            bmi = st.number_input(
                "⚖️ BMI", min_value=0.0, max_value=70.0, value=28.0, step=0.1
            )
            dpf = st.number_input(
                "🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.001, format="%.3f"
            )
            age = st.number_input(
                "🎂 Age (years)", min_value=1, max_value=120, value=35, step=1
            )

        submitted = st.form_submit_button("🔍 Predict", type="primary")

    if submitted:
        input_data = [
            int(pregnancies),
            float(glucose),
            float(blood_pressure),
            float(skin_thickness),
            float(insulin),
            float(bmi),
            float(dpf),
            int(age),
        ]

        try:
            is_diabetic, risk_score = diabetes_prediction(model, input_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.subheader("Result")

        if is_diabetic:
            st.markdown(
                '<div class="result-card result-bad">⚠️ The person is DIABETIC.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card result-ok">✅ The person is NOT diabetic.</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.write(f"**Diabetes risk score: {risk_score * 100:.1f}%**")
        st.progress(float(min(max(risk_score, 0.0), 1.0)))
        st.caption(
            "The score comes from the SVM's decision value (above 50% means diabetic). "
            "It shows how far the input is from the decision boundary and is not a "
            "calibrated probability."
        )

        with st.expander("View the values you entered"):
            st.dataframe(
                pd.DataFrame([input_data], columns=feature_names),
                hide_index=True,
            )

        st.caption("For educational purposes only, not a medical diagnosis.")


if __name__ == "__main__":
    main()
