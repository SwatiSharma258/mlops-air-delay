# ingest.py
# ================================================
# Data Ingestion Script for Air Traffic Delay Prediction
# ================================================

import os
import pandas as pd
import numpy as np

# Paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

# -----------------------------------------------
# Step 1: Create folders if they don’t exist
# -----------------------------------------------
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# -----------------------------------------------
# Step 2: Simulate raw flight & weather data
# -----------------------------------------------
def generate_fake_data(n_rows=5000):
    """Generate fake flight + weather dataset for demonstration."""
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
        "day_of_week": np.random.randint(1, 8, n_rows),
        "delay_minutes": np.random.randint(0, 120, n_rows)
    }
    
    df = pd.DataFrame(data)
    return df

# -----------------------------------------------
# Step 3: Clean the data
# -----------------------------------------------
def clean_data(df: pd.DataFrame):
    """Simple cleaning logic to remove invalid rows."""
    df = df.dropna()
    df = df[df["visibility"] > 0]
    df = df[df["wind_speed"] >= 0]
    return df

# -----------------------------------------------
# Step 4: Main Ingestion Function
# -----------------------------------------------
def ingest_data():
    print("🚀 Starting data ingestion process...")

    # Generate fake dataset
    raw_df = generate_fake_data()

    # Save raw file
    raw_file_path = os.path.join(RAW_DATA_PATH, "flight_weather_raw.csv")
    raw_df.to_csv(raw_file_path, index=False)
    print(f"✅ Raw data saved at: {raw_file_path}")

    # Clean and save processed file
    processed_df = clean_data(raw_df)
    processed_file_path = os.path.join(PROCESSED_DATA_PATH, "flight_weather_clean.csv")
    processed_df.to_csv(processed_file_path, index=False)
    print(f"✅ Processed data saved at: {processed_file_path}")

    print("🎯 Data ingestion completed successfully!")
    return processed_df


# -----------------------------------------------
# Run when executed directly
# -----------------------------------------------
if __name__ == "__main__":
    ingest_data()
