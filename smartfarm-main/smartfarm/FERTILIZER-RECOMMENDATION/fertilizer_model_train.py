import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# ✅ Load dataset
df = pd.read_csv("fertilizer_data.csv")

# ✅ Clean column names: lowercase, remove spaces
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Check required columns
required_cols = ["soil_type", "crop_type", "fertilizer_name"]
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"❌ Expected columns not found in CSV: {missing}")

# ✅ One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])

# ✅ Prepare features and target
X = df_encoded.drop("fertilizer_name", axis=1)
y = df_encoded["fertilizer_name"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Save model and feature names
model_package = {
    "model": model,
    "features": X.columns.tolist()
}
with open("fertilizer_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("✅ Model trained and saved as fertilizer_model.pkl")
