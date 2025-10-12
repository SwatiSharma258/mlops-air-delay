# ingest.py
# ================================================
# Data Ingestion Script for Air Traffic Delay Prediction
# ================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

# -------------------------------
# Paths (configurable)
# -------------------------------
RAW_DATA_PATH = Path("data/raw/")
PROCESSED_DATA_PATH = Path("data/processed/")
RAW_FILE_NAME = "flight_weather_raw.csv"
PROCESSED_FILE_NAME = "flight_weather_clean.csv"

# Ensure folders exist
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Generate fake dataset (demo purpose)
# -------------------------------
def generate_fake_data(n_rows: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic flight + weather dataset for demo/testing purposes.
    """
    np.random.seed(42)
    airports = ["ATL", "ORD", "DFW", "LAX", "JFK", "SFO"]

    data = {
        "flight_id": range(1, n_rows + 1),
        "departure_airport": np.random.choice(airports, n_rows),
        "arrival_airport": np.random.choice(airports, n_rows),
        "departure_time": np.random.randint(0, 1440, n_rows),  # minutes of day
        "temperature": np.random.uniform(10, 35, n_rows),
        "wind_speed": np.random.uniform(0, 25, n_rows),
        "visibility": np.random.uniform(1, 10, n_rows),
        "day_of_week": np.random.randint(0, 7, n_rows),
        "delay_minutes": np.random.randint(0, 120, n_rows)
    }

    df = pd.DataFrame(data)
    return df

# -------------------------------
# Clean the dataset
# -------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning: remove nulls, invalid weather values.
    """
    df = df.dropna()
    df = df[df["visibility"] > 0]
    df = df[df["wind_speed"] >= 0]
    return df

# -------------------------------
# Ingest data (main function)
# -------------------------------
def ingest_data(real_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Ingest raw flight/weather data, clean it, and save processed CSV.

    Parameters:
    -----------
    real_csv : Optional[str]
        Path to real CSV data. If None, generates fake data.

    Returns:
    --------
    pd.DataFrame
        Processed dataframe ready for training.
    """
    print("🚀 Starting data ingestion process...")

    # Load or generate data
    if real_csv and os.path.exists(real_csv):
        raw_df = pd.read_csv(real_csv)
        print(f"✅ Loaded real data from {real_csv}")
    else:
        raw_df = generate_fake_data()
        print("ℹ️ Generated fake dataset for demo/testing purposes")

    # Save raw data
    raw_file_path = RAW_DATA_PATH / RAW_FILE_NAME
    raw_df.to_csv(raw_file_path, index=False)
    print(f"✅ Raw data saved at: {raw_file_path}")

    # Clean data
    processed_df = clean_data(raw_df)

    # Save processed data
    processed_file_path = PROCESSED_DATA_PATH / PROCESSED_FILE_NAME
    processed_df.to_csv(processed_file_path, index=False)
    print(f"✅ Processed data saved at: {processed_file_path}")

    print("🎯 Data ingestion completed successfully!")
    return processed_df

# -------------------------------
# Run directly
# -------------------------------
if __name__ == "__main__":
    ingest_data()
