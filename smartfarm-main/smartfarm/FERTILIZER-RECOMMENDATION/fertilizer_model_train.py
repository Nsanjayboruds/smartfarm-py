import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("fertilizer_data.csv")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])

X = df_encoded.drop("fertilizer", axis=1)
y = df_encoded["fertilizer"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open("fertilizer_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as fertilizer_model.pkl")
