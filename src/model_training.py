import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Activation, Dropout

# Define dataset path
DATASET_PATH = r"D:\Jio-AIDS\Q3\DL\Proj\dataset_features"

def load_data():
    """Loads the pre-extracted MFCC features and labels."""
    X = joblib.load(os.path.join(DATASET_PATH, "X.joblib"))
    y = joblib.load(os.path.join(DATASET_PATH, "y.joblib"))
    print(f"Dataset Loaded: X shape: {X.shape}, y shape: {y.shape}")
    return X, y

# Load dataset
X, y = load_data()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Reshape for CNN
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

# Decision Tree Model
def train_decision_tree():
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    return dtree

# CNN Model
def train_cnn():
    """Defines and trains the CNN model."""
    model = Sequential([
        Conv1D(64, 5, padding='same', input_shape=(40, 1)),
        Activation('relu'),
        Dropout(0.1),
        MaxPooling1D(pool_size=4),
        
        Conv1D(128, 5, padding='same'),
        Activation('relu'),
        Dropout(0.1),
        MaxPooling1D(pool_size=4),
        
        Conv1D(256, 5, padding='same'),
        Activation('relu'),
        Dropout(0.1),
        Flatten(),
        
        Dense(8),
        Activation('softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
                  metrics=['accuracy'])

    history = model.fit(X_train_cnn, y_train, batch_size=16, epochs=200, validation_data=(X_test_cnn, y_test))

    return model, history  # Returns trained model and training history
