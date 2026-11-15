
"""DermaAI - AI-Powered Skin Disease Detection and Analysis System."""

import streamlit as st
from PIL import Image

from utils.predictor import SkinConditionPredictor
from utils.gemini_helper import GeminiExplainer


# Page configuration
st.set_page_config(
    page_title="DermaAI",
    page_icon="🔬",
    layout="wide"
)


# Load models only once
@st.cache_resource
def load_predictor():
    return SkinConditionPredictor()


@st.cache_resource
def load_gemini():
    return GeminiExplainer()


# Application header
st.title("🔬 DermaAI")
st.subheader("AI-Powered Skin Disease Detection and Analysis System")

st.write(
    "Upload a photograph of a skin condition to receive "
    "an AI-generated prediction and educational information."
)

st.info(
    "DermaAI is an educational tool and does not provide "
    "medical diagnoses. Always consult a qualified dermatologist."
)


# Image uploader
st.divider()

st.header("Upload Skin Image")

uploaded_file = st.file_uploader(
    "Choose a skin photograph",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photograph in JPG or PNG format."
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Skin Photograph",
        width=450
    )

    if st.button("Analyze Image", type="primary"):

        try:
            with st.spinner("Analyzing image..."):

                predictor = load_predictor()
                result = predictor.predict(image)

            st.session_state["prediction"] = result
            st.session_state["gemini_explanation"] = None

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# Display prediction
if st.session_state.get("prediction") is not None:

    result = st.session_state["prediction"]

    st.divider()
    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Skin Condition",
            result["condition"]
        )

    with col2:
        st.metric(
            "Model Confidence",
            f"{result['confidence']}%"
        )

    st.caption(
        "Confidence is the model's softmax score, "
        "not a medically calibrated probability."
    )

    st.warning(
        "This prediction is preliminary and does not "
        "confirm that you have this skin condition."
    )

    st.divider()
    st.header("Educational Information")

    if st.button("Generate AI Explanation"):

        try:
            with st.spinner("Gemini is preparing your explanation..."):

                explainer = load_gemini()

                explanation = explainer.explain_condition(
                    result["condition"]
                )

                st.session_state["gemini_explanation"] = explanation

        except Exception as e:
            st.error(f"Gemini explanation failed: {e}")

    if st.session_state.get("gemini_explanation"):

        st.markdown(
            st.session_state["gemini_explanation"]
        )


# Project team section
st.divider()

st.header("👥 Meet Our Team")
st.write("**Group 2 | DermaAI Project**")

team_members = [
    ("Armaan Alam", "2306188"),
    ("Soumyajit Paul", "2306317"),
    ("Ipshita Tewari", "2306285"),
    ("Soumyadeep Mohanta", "2306315"),
    ("Parnabi Bhattacharjee", "2406604"),
    ("Soumik Pramanik", "2306313"),
    
]

col1, col2 = st.columns(2)

for index, (name, registration) in enumerate(team_members):

    with col1 if index % 2 == 0 else col2:

        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(f"Roll Number: {registration}")


# Footer
st.divider()

st.caption(
    "DermaAI | 7th Semester Major Project | Group 2"
)

st.caption(
    "For educational purposes only. "
    "Not a substitute for professional medical advice."
)

