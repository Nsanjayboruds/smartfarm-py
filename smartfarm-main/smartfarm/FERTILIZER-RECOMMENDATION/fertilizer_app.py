# fertilizer_app.py

import streamlit as st
import pandas as pd
import pickle
import os

# Load model and expected features
model_path = os.path.join(os.path.dirname(__file__), "fertilizer_model.pkl")
with open(model_path, "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
feature_names = model_package["features"]

st.title("🌿 Fertilizer Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
temperature = st.slider("Temperature (°C)", 10, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 70)
moisture = st.slider("Soil Moisture (%)", 0, 100, 30)

soil = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Black", "Red", "Alluvial"])
crop = st.selectbox("Crop Type", [
    "Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", 
    "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"
])

if st.button("🌱 Recommend Fertilizer"):
    # Create input dictionary with all expected features
    input_dict = dict.fromkeys(feature_names, 0)

    # Update numerical features
    input_dict["temperature"] = temperature
    input_dict["humidity"] = humidity
    input_dict["moisture"] = moisture
    input_dict["nitrogen"] = N
    input_dict["potassium"] = K
    input_dict["phosphorous"] = P

    # Update categorical one-hot fields
    soil_key = f"soil_type_{soil.strip().lower()}"
    crop_key = f"crop_type_{crop.strip().lower().replace(' ', '_')}"

    if soil_key in input_dict:
        input_dict[soil_key] = 1
    if crop_key in input_dict:
        input_dict[crop_key] = 1

    # Convert to DataFrame with expected column order
    input_df = pd.DataFrame([input_dict])[feature_names]

    # Predict fertilizer
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Recommended Fertilizer: **{prediction}**")





