# model.py
# ================================================
# Model Training and Loading Utilities for Air Delay Prediction
# ================================================

import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from typing import Tuple
import numpy as np
import pandas as pd


def create_and_train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: str = "models/air_delay_model.pkl"
) -> RandomForestRegressor:
    """
    Train a RandomForest model for flight delay prediction and log metrics to MLflow.

    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    model_path : str
        Path to save the trained model

    Returns:
    --------
    Trained RandomForestRegressor
    """

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Start MLflow run
    with mlflow.start_run(run_name=f"AirDelayModel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_Score", r2)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        joblib.dump(model, model_path)
        print(f"✅ Model trained and saved at: {model_path}")
        print(f"📊 Metrics: MAE={mae:.3f}, MSE={mse:.3f}, R2={r2:.3f}")

    return model


def load_model(model_path: str = "models/air_delay_model.pkl") -> RandomForestRegressor:
    """
    Load a trained model from disk.

    Parameters:
    -----------
    model_path : str
        Path to the trained model file

    Returns:
    --------
    RandomForestRegressor
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
    return model
