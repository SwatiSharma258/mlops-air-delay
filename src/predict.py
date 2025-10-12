# predict.py
# ================================================
# Standalone Prediction Script for Air Traffic Delay Prediction
# ================================================

import joblib
import numpy as np
import pandas as pd
from feature import extract_features

# -----------------------------------------------
# Step 1: Load model
# -----------------------------------------------
MODEL_PATH = "models/flight_delay_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except:
    raise FileNotFoundError("❌ Model file not found. Please run train_model.py first.")

# -----------------------------------------------
# Step 2: Sample input data
# -----------------------------------------------
sample_input = {
    "departure_time": "14:30",
    "arrival_airport": "JFK",
    "departure_airport": "ATL",
    "temperature": 27.5,
    "wind_speed": 10.2,
    "visibility": 8.5,
    "day_of_week": 5
}

print("\n🛫 Input Flight Data:")
for k, v in sample_input.items():
    print(f"  {k}: {v}")

# -----------------------------------------------
# Step 3: Feature extraction
# -----------------------------------------------
features = extract_features(sample_input)
features = np.array(features).reshape(1, -1)

# -----------------------------------------------
# Step 4: Predict delay
# -----------------------------------------------
prediction = model.predict(features)[0]

print("\n🎯 Predicted Flight Delay:")
print(f"   ✈️ Estimated delay: {prediction:.2f} minutes")
