import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Fertilizer Recommender", page_icon="🌿", layout="centered")

# Optional custom CSS to tweak UI
st.markdown("""
    <style>
    .stSlider > div[data-baseweb="slider"] {
        padding: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Fertilizer Recommendation System")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "fertilizer_model.pkl")
with open(model_path, "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
feature_names = model_package["features"]

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=40)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=41)

temperature = st.slider("Temperature (°C)", 10, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 53)
moisture = st.slider("Soil Moisture (%)", 0, 100, 14)

soil = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Black", "Red", "Alluvial"])
crop = st.selectbox("Crop Type", ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Wheat"])

# Recommend fertilizer
if st.button("Recommend Fertilizer"):
    # Prepare input
    input_dict = dict.fromkeys(feature_names, 0)
    input_dict.update({
        "nitrogen": N,
        "phosphorous": P,
        "potassium": K,
        "temparature": temperature,
        "humidity": humidity,
        "moisture": moisture,
        f"soil_type_{soil.lower()}": 1,
        f"crop_type_{crop.lower().replace(' ', '_')}": 1
    })

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
