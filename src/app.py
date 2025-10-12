# app.py
# ================================================
# Real-Time Air Traffic Delay Prediction API
# ================================================

from flask import Flask, request, jsonify
import numpy as np
import joblib
from feature import extract_features

app = Flask(__name__)

# -------------------------------
# Load trained model
# -------------------------------
try:
    model = joblib.load("models/flight_delay_model.pkl")
except:
    model = None
    print("⚠️ Model not found! Using mock predictions for demo.")

# -------------------------------
# Home route
# -------------------------------
@app.route('/')
def home():
    return jsonify({
        "message": "✈️ Real-Time Air Traffic Delay Prediction API is running!",
        "status": "success",
        "routes": ["/predict"]
    })

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Input JSON:
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
    data = request.get_json()

    # Extract numerical features
    features = extract_features(data)

    if model:
        prediction = model.predict([features])[0]
    else:
        # Mock delay prediction
        prediction = np.random.randint(0, 60)  # fake minutes delayed

    result = {
        "predicted_delay_minutes": float(prediction),
        "message": "Prediction successful"
    }

    return jsonify(result)

# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
