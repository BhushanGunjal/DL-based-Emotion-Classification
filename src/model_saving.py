import os
import joblib
from tensorflow import keras
from config import MODEL_DIR


# Ensure the directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Save models
def save_models(dtree, cnn_model):
    """Saves the trained models."""
    try:
        joblib.dump(dtree, os.path.join(MODEL_DIR, "decision_tree.pkl"))
        cnn_model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))
        print("✅ Models Saved Successfully!")
    except Exception as e:
        print(f"❌ Error while saving models: {e}")

# Load models
def load_models():
    """Loads saved models, or returns None if not found."""
    dtree_path = os.path.join(MODEL_DIR, "decision_tree.pkl")
    cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")

    dtree, cnn_model = None, None

    # Load Decision Tree Model
    if os.path.exists(dtree_path):
        dtree = joblib.load(dtree_path)
        print("✅ Decision Tree Model Loaded!")
    else:
        print("⚠️ Decision Tree Model Not Found!")

    # Load CNN Model
    if os.path.exists(cnn_path):
        cnn_model = keras.models.load_model(cnn_path)
        print("✅ CNN Model Loaded!")
    else:
        print("⚠️ CNN Model Not Found!")

    return dtree, cnn_model
