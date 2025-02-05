from src.model_training import train_decision_tree, train_cnn, X_test, y_test, X_test_cnn
from src.model_saving import save_models, load_models
from src.evaluation import evaluate_model
from src.visualize_results import plot_training_results
import os
from config import MODEL_DIR



dtree_path = os.path.join(MODEL_DIR, "decision_tree.pkl")
cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")


print("🔹 Checking if models exist...")



# Train only if models do not exist
if not os.path.exists(dtree_path) or not os.path.exists(cnn_path):
    print("🔹 Models not found! Training new models...")

    # Train models
    dtree = train_decision_tree()
    print("✅ Decision Tree trained.")

    cnn_model, history = train_cnn()
    print("✅ CNN Model trained.")

    # Save models
    print("🔹 Saving models...")
    save_models(dtree, cnn_model)
    print("✅ Models saved.")
else:
    print("✅ Models already exist. Skipping training.")




# Load models
print("🔹 Loading models...")
dtree, cnn_model = load_models()
print("✅ Models loaded.")

# Evaluate models
if dtree:
    print("🔹 Evaluating Decision Tree...")
    evaluate_model(dtree, X_test, y_test, "DecisionTree")
    print("✅ Decision Tree evaluated.")

if cnn_model:
    print("🔹 Evaluating CNN...")
    evaluate_model(cnn_model, X_test_cnn, y_test, "CNN")
    print("✅ CNN evaluated.")




















# Plot training history ONLY if a new model was trained
if history is not None:
    print("🔹 Plotting training history...")
    plot_training_results(history)
    print("✅ Training history plotted.")

print("🚀 All tasks completed successfully!")
