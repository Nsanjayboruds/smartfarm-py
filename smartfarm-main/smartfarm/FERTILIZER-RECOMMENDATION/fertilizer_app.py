import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="SmartFarm - Fertilizer", page_icon="🌿")

st.title("🌿 Fertilizer Recommendation System")

# ✅ Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "fertilizer_data.csv")
if not os.path.exists(csv_path):
    st.error("❌ 'fertilizer_data.csv' not found!")
    st.stop()

df = pd.read_csv(csv_path)

# ✅ Encode data
df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])
X = df_encoded.drop("fertilizer", axis=1)
y = df_encoded["fertilizer"]

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Load or retrain model
model_path = os.path.join(os.path.dirname(__file__), "fertilizer_model.pkl")
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.warning(f"⚠️ Model loading failed: {e}. Retraining...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

# ✅ User Inputs
soil_types = df["soil_type"].unique().tolist()
crop_types = df["crop_type"].unique().tolist()

N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
temperature = st.slider("Temperature (°C)", 10, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 70)
moisture = st.slider("Soil Moisture (%)", 0, 100, 30)

soil = st.selectbox("Soil Type", soil_types)
crop = st.selectbox("Crop Type", crop_types)

# ✅ Prediction
if st.button("Recommend Fertilizer"):
    input_dict = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "moisture": moisture,
    }

    # One-hot encode soil and crop
    for s_col in df_encoded.columns:
        if "soil_type_" in s_col:
            input_dict[s_col] = 1 if s_col == f"soil_type_{soil}" else 0
        if "crop_type_" in s_col:
            input_dict[s_col] = 1 if s_col == f"crop_type_{crop}" else 0

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Recommended Fertilizer: **{prediction}**")

# Optional: show sample data
with st.expander("📊 View Dataset Sample"):
    st.dataframe(df.head())
