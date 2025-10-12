# app.py
# ================================================
# Real-Time Air Traffic Delay Prediction API
# ================================================

from flask import Flask, request, jsonify
import numpy as np
import joblib
from typing import Dict, Any
from feature import extract_features

app = Flask(__name__)

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "models/flight_delay_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"⚠️ Model not found at {MODEL_PATH}. Using mock predictions for demo.")

# -------------------------------
# Home route
# -------------------------------
@app.route("/", methods=["GET"])
def home() -> Dict[str, Any]:
    return jsonify({
        "message": "✈️ Real-Time Air Traffic Delay Prediction API is running!",
        "status": "success",
        "routes": ["/predict"]
    })

# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict() -> Dict[str, Any]:
    """
    Expects JSON input:
    {
        "departure_time": "14:30",
        "arrival_airport": "JFK",
        "departure_airport": "ATL",
        "temperature": 25.4,
        "wind_speed": 12.3,
        "visibility": 10.0,
        "day_of_week": 4
    }
    """
    try:
        data: Dict[str, Any] = request.get_json()
        if not data:
            raise ValueError("No JSON payload provided")

        # Extract features using your feature engineering function
        features = extract_features(data)

        if model:
            prediction = model.predict([features])[0]
        else:
            # Mock prediction for demo
            prediction = np.random.randint(0, 60)

        result = {
            "predicted_delay_minutes": float(prediction),
            "message": "Prediction successful"
        }

    except Exception as e:
        result = {
            "predicted_delay_minutes": None,
            "message": f"Prediction failed: {str(e)}"
        }

    return jsonify(result)

# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
