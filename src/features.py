# feature.py
# ================================================
# Feature Engineering for Air Traffic Delay Prediction
# ================================================

import pandas as pd
import numpy as np
from datetime import datetime

# --------------------------------
# Helper: Convert HH:MM → Minutes
# --------------------------------
def time_to_minutes(time_str):
    try:
        t = datetime.strptime(time_str, "%H:%M")
        return t.hour * 60 + t.minute
    except:
        return np.nan

# --------------------------------
# Main Feature Extraction Function
# --------------------------------
def extract_features(data: dict):
    """
    Converts raw API input into model-ready features.
    
    Expected input format:
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

    # Convert to DataFrame for consistency
    df = pd.DataFrame([data])

    # Convert time to numeric
    df["departure_mins"] = df["departure_time"].apply(time_to_minutes)

    # Encode categorical variables (simple label encoding)
    airport_map = {"ATL": 1, "ORD": 2, "DFW": 3, "LAX": 4, "JFK": 5, "SFO": 6}
    df["arrival_airport_code"] = df["arrival_airport"].map(airport_map).fillna(0)
    df["departure_airport_code"] = df["departure_airport"].map(airport_map).fillna(0)

    # Drop unused text columns
    df = df.drop(["departure_time", "arrival_airport", "departure_airport"], axis=1)

    # Fill missing values
    df = df.fillna(0)

    # Return feature array
    return df.values[0]
