import joblib
import os

# Define paths
DATASET_PATH = r"D:\Jio-AIDS\Q3\DL\Proj\dataset_features"

def load_data():
    """Loads the pre-extracted MFCC features and labels."""
    X = joblib.load(os.path.join(DATASET_PATH, "X.joblib"))
    y = joblib.load(os.path.join(DATASET_PATH, "y.joblib"))
    print(f"Dataset Loaded: X shape: {X.shape}, y shape: {y.shape}")
    return X, y
