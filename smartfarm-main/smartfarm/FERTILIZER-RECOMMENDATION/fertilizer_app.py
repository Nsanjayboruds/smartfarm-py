import streamlit as st
import pandas as pd
import pickle

# Load the model and feature names
with open("fertilizer_model.pkl", "rb") as f:
    model_package = pickle.load(f)
    model = model_package["model"]
    feature_names = model_package["features"]

# Streamlit app title
st.title("🌿 Fertilizer Recommendation System")

# Dropdown options
soil_types = ["Loamy", "Clayey", "Sandy", "Black", "Red"]
crop_types = ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
              "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"]

# User inputs
temperature = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
moisture = st.slider("Moisture (%)", 0, 100, 40)
nitrogen = st.slider("Nitrogen content (N)", 0, 100, 25)
phosphorous = st.slider("Phosphorous content (P)", 0, 100, 25)
potassium = st.slider("Potassium content (K)", 0, 100, 25)
soil_type = st.selectbox("Soil Type", soil_types)
crop_type = st.selectbox("Crop Type", crop_types)

# Predict button
if st.button("Predict Fertilizer"):
    # Create default input dictionary with all features set to 0
    input_dict = dict.fromkeys(feature_names, 0)

    # Fill in numerical features
    input_dict["temparature"] = temperature
    input_dict["humidity"] = humidity
    input_dict["moisture"] = moisture
    input_dict["nitrogen"] = nitrogen
    input_dict["phosphorous"] = phosphorous
    input_dict["potassium"] = potassium

    # One-hot encoding for soil and crop
    soil_feature = f"soil_type_{soil_type.lower()}"
    crop_feature = f"crop_type_{crop_type.lower().replace(' ', '_')}"

    if soil_feature in input_dict:
        input_dict[soil_feature] = 1
    if crop_feature in input_dict:
        input_dict[crop_feature] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Prediction
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
