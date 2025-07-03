import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and clean dataset
df = pd.read_csv("fertilizer_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Encode categorical columns
df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])

X = df_encoded.drop("fertilizer_name", axis=1)
y = df_encoded["fertilizer_name"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and feature names
model_package = {
    "model": model,
    "features": X.columns.tolist()
}

with open("fertilizer_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("✅ Model and feature names saved to fertilizer_model.pkl")
