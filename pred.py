import os
import numpy as np
import librosa
from tensorflow import keras
from config import MODEL_DIR  # Import model path from config

class LivePredictions:
    """
    This class is used for performing live predictions on audio files.
    """

    def __init__(self, model_name, audio_file):
        """
        Initializes the model path and input audio file.
        """
        self.model_path = os.path.join(MODEL_DIR, model_name)
        self.audio_file = audio_file
        self.loaded_model = None

    def load_model(self):
        """
        Loads the pre-trained CNN model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model file not found at {self.model_path}")

        print(f"🔹 Loading model from: {self.model_path}")
        self.loaded_model = keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully!")

    def make_predictions(self):
        """
        Processes the audio file, extracts MFCC features, and makes a prediction.
        """
        if not os.path.exists(self.audio_file):
            raise FileNotFoundError(f"❌ Audio file not found: {self.audio_file}")

        print(f"🔹 Processing audio file: {self.audio_file}")
        
        # Load and preprocess audio file
        data, sampling_rate = librosa.load(self.audio_file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        
        # Reshape input for the CNN model
        x = np.expand_dims(mfccs, axis=1)  # Add feature dimension
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        # Predict the emotion
        predictions = self.loaded_model.predict(x)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Convert class index to emotion label
        emotion_label = self.convert_class_to_emotion(predicted_class)

        print(f"🎯 Prediction: {emotion_label}")
        return emotion_label

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Converts model's integer output to a human-readable emotion label.
        """
        label_conversion = {
            0: 'neutral',
            1: 'calm',
            2: 'happy',
            3: 'sad',
            4: 'angry',
            5: 'fearful',
            6: 'disgust',
            7: 'surprised'
        }

        return label_conversion.get(pred, "Unknown")

# Example Usage
if __name__ == "__main__":
    model_filename = "cnn_model.h5"  # Change if needed
    audio_filename = "sample_audio.wav"  # Replace with your test audio file

    pred = LivePredictions(model_filename, audio_filename)
    pred.load_model()
    pred.make_predictions()
