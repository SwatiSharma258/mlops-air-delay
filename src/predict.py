# predict.py
# ================================================
# Standalone Prediction Script for Air Traffic Delay Prediction
# ================================================

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from feature import extract_features

MODEL_PATH = "models/flight_delay_model.pkl"

# -----------------------------------------------
# Load trained model
# -----------------------------------------------
def load_model(model_path: str = MODEL_PATH):
    """Load the trained ML model from disk."""
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Model not found at {model_path}. Please run train.py first.")

# -----------------------------------------------
# Predict function
# -----------------------------------------------
def predict_delay(model, input_data: Union[Dict[str, Any], pd.DataFrame]) -> np.ndarray:
    """
    Predict flight delays given input data.

    Parameters:
    -----------
    model : scikit-learn model
        Trained model
    input_data : dict or pd.DataFrame
        Single input dictionary or DataFrame with multiple rows

    Returns:
    --------
    np.ndarray
        Predicted delays in minutes
    """
    if isinstance(input_data, dict):
        features = extract_features(input_data).reshape(1, -1)
    elif isinstance(input_data, pd.DataFrame):
        features = np.array([extract_features(row) for _, row in input_data.iterrows()])
    else:
        raise ValueError("Input must be a dict or pandas DataFrame")

    return model.predict(features)

# -----------------------------------------------
# Main execution (demo)
# -----------------------------------------------
if __name__ == "__main__":
    model = load_model()

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

    prediction = predict_delay(model, sample_input)[0]

    print("\n🎯 Predicted Flight Delay:")
    print(f"   ✈️ Estimated delay: {prediction:.2f} minutes")
