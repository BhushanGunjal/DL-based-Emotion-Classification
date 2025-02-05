from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, X, y, model_type="Model"):
    """Evaluates a trained model with test data."""
    predictions = model.predict(X)

        # Convert probabilities to class labels (only for CNN)
    if len(predictions.shape) > 1:  # CNN outputs probabilities
        predictions = np.argmax(predictions, axis=1)

    print(f"\n🔹 Classification Report for {model_type}:")
    print(classification_report(y, predictions))
    print(f"\n🔹 Confusion Matrix for {model_type}:")
    print(confusion_matrix(y, predictions))
