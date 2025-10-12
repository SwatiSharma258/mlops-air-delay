# feature.py
# ================================================
# Feature Engineering for Air Traffic Delay Prediction
# ================================================

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

# --------------------------------
# Helper: Convert HH:MM → Minutes
# --------------------------------
def time_to_minutes(time_str: str) -> float:
    """Convert HH:MM string to total minutes. Returns NaN if invalid."""
    try:
        t = datetime.strptime(time_str, "%H:%M")
        return t.hour * 60 + t.minute
    except Exception:
        return np.nan

# --------------------------------
# Main Feature Extraction Function
# --------------------------------
def extract_features(data: Dict[str, Any]) -> np.ndarray:
    """
    Convert raw API input dictionary into a model-ready numeric feature array.
    
    Parameters:
    -----------
    data : dict
        Input dictionary with keys:
        - departure_time (str, "HH:MM")
        - arrival_airport (str)
        - departure_airport (str)
        - temperature (float)
        - wind_speed (float)
        - visibility (float)
        - day_of_week (int, 0=Monday, 6=Sunday)
    
    Returns:
    --------
    np.ndarray
        1D array of numeric features ready for model prediction.
    """
    
    df = pd.DataFrame([data])

    # Convert time to numeric minutes
    df["departure_mins"] = df["departure_time"].apply(time_to_minutes)

    # Encode categorical airports with safe mapping
    airport_map = {"ATL": 1, "ORD": 2, "DFW": 3, "LAX": 4, "JFK": 5, "SFO": 6}
    df["arrival_airport_code"] = df["arrival_airport"].map(airport_map).fillna(0)
    df["departure_airport_code"] = df["departure_airport"].map(airport_map).fillna(0)

    # Drop original text columns
    df = df.drop(columns=["departure_time", "arrival_airport", "departure_airport"], errors='ignore')

    # Fill missing numerical values
    df = df.fillna(0)

    # Return as 1D numpy array
    return df.values[0]
