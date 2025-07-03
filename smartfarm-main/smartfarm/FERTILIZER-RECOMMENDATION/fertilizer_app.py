import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="SmartFarm - Fertilizer", page_icon="🌿")
st.title("🌿 Fertilizer Recommendation System")

# Load CSV data
csv_path = os.path.join(os.path.dirname(__file__), "fertilizer_data.csv")
if not os.path.exists(csv_path):
    st.error("❌ 'fertilizer_data.csv' not found!")
    st.stop()

df = pd.read_csv(csv_path)
st.write("Columns in CSV:", df.columns.tolist())

# Verify the columns for one-hot encoding
# Adjust these names to match your CSV header exactly
expected_cols = ["soil_type", "crop_type"]
missing = [col for col in expected_cols if col not in df.columns]
if missing:
    st.error(f"Expected columns not found in CSV: {missing}")
    st.stop()
df_encoded = pd.get_dummies(df, columns=["Soil Type", "Crop Type"])


X = df_encoded.drop("fertilizer", axis=1)
y = df_encoded["fertilizer"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train model
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

# User inputs (adjust these if needed to match your training schema)
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
temperature = st.slider("Temperature (°C)", 10, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 70)
moisture = st.slider("Soil Moisture (%)", 0, 100, 30)

soil = st.selectbox("Soil Type", df["soil_type"].unique().tolist())
crop = st.selectbox("Crop Type", df["crop_type"].unique().tolist())

if st.button("Recommend Fertilizer"):
    input_dict = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "moisture": moisture,
    }
    # Set one-hot encoding for soil and crop types based on the encoded columns in df_encoded
    for col in df_encoded.columns:
        if col.startswith("soil_type_"):
            input_dict[col] = 1 if col == f"soil_type_{soil}" else 0
        if col.startswith("crop_type_"):
            input_dict[col] = 1 if col == f"crop_type_{crop}" else 0

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
