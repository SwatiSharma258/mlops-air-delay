# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# Paths
DATA_PATH = "data/raw/data.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

def load_data(path=DATA_PATH):
    """Load raw data from CSV file."""
    df = pd.read_csv(path)
    print(f"Data loaded from {path}. Shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the dataset: handle missing values, encode categorical, scale numeric."""
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(df)
    
    # Save the preprocessor for future use
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")
    
    return X_processed

def save_processed_data(X, path=PROCESSED_PATH):
    """Save processed data to CSV."""
    df_processed = pd.DataFrame(X)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_processed.to_csv(path, index=False)
    print(f"Processed data saved to {path}")

if __name__ == "__main__":
    df = load_data()
    X_processed = preprocess_data(df)
    save_processed_data(X_processed)
