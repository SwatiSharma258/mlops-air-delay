# train.py
# ================================================
# End-to-End Model Training Pipeline for
# Real-Time Air Traffic Delay Prediction
# ================================================

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

from ingest import ingest_data  # Import data ingestion function

# -----------------------------------------------
# Step 1: Folder setup
# -----------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -----------------------------------------------
# Step 2: Run data ingestion
# -----------------------------------------------
print("📥 Running data ingestion...")
df = ingest_data()
print(f"✅ Ingested {len(df)} rows of data for training.\n")

# -----------------------------------------------
# Step 3: Prepare training data
# -----------------------------------------------
X = df.drop(columns=["delay_minutes"])
y = df["delay_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# Step 4: Initialize MLflow tracking
# -----------------------------------------------
mlflow.set_experiment("Air_Traffic_Delay_Prediction")

with mlflow.start_run():
    print("🚀 Training Linear Regression model...")

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters & metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2_Score", r2)

    # Save model locally
    model_path = "models/flight_delay_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    print(f"""
    📊 Model Performance:
    ----------------------
    MAE  = {mae:.2f}
    MSE  = {mse:.2f}
    R2   = {r2:.2f}
    """)

print("End-to-end training completed successfully!")
