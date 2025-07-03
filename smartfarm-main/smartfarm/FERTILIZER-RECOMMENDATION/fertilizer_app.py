import streamlit as st
import pandas as pd
import pickle

# Load model and features
with open("fertilizer_model.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
feature_names = model_package["features"]

# Dropdown options
soil_types = ["Loamy", "Clayey", "Sandy", "Black", "Red"]
crop_types = ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"]

# Streamlit UI
st.title("🌿 Fertilizer Recommendation System")

temperature = st.slider("Temperature (°C)", 0, 50, 30)
humidity = st.slider("Humidity (%)", 0, 100, 50)
moisture = st.slider("Moisture (%)", 0, 100, 40)
soil_type = st.selectbox("Soil Type", soil_types)
crop_type = st.selectbox("Crop Type", crop_types)
nitrogen = st.slider("Nitrogen", 0, 100, 20)
potassium = st.slider("Potassium", 0, 100, 20)
phosphorous = st.slider("Phosphorous", 0, 100, 20)

if st.button("Get Recommendation"):
    # Prepare input
    input_dict = {
        "temparature": temperature,
        "humidity": humidity,
        "moisture": moisture,
        "nitrogen": nitrogen,
        "potassium": potassium,
        "phosphorous": phosphorous,
        f"soil_type_{soil_type.lower()}": 1,
        f"crop_type_{crop_type.lower().replace(' ', '_')}": 1
    }

    # Fill in missing features with 0
    input_data = {feature: input_dict.get(feature, 0) for feature in feature_names}
    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
