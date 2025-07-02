import streamlit as st
import numpy as np
import pickle
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "fertilizer_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🌿 Fertilizer Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
temperature = st.slider("Temperature (°C)", 10, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 70)
moisture = st.slider("Soil Moisture (%)", 0, 100, 30)

soil = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Black", "Red", "Alluvial"])
crop = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Millets"])

if st.button("Recommend Fertilizer"):
    # One-hot encode soil and crop
    soil_types = ["Loamy", "Clay", "Sandy", "Black", "Red", "Alluvial"]
    crop_types = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Millets"]

    soil_encoded = [1 if s == soil else 0 for s in soil_types]
    crop_encoded = [1 if c == crop else 0 for c in crop_types]

    input_features = np.array([[N, P, K, temperature, humidity, moisture] + soil_encoded + crop_encoded])
    prediction = model.predict(input_features)[0]

    st.success(f"🌱 Recommended Fertilizer: **{prediction}**")



