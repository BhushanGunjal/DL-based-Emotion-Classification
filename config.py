import os

# Define the paths for training files and TESS dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use raw string (r"") to prevent escape sequence issues
TRAINING_FILES_PATH = r"data\\" 
TESS_ORIGINAL_FOLDER_PATH = r"temp\dataverse_files"
SAVE_DIR_PATH = r"dataset_features\\"
MODEL_DIR = r"D:\Jio-AIDS\Q3\DL\Proj\models"

# Ensure directories exist
os.makedirs(TRAINING_FILES_PATH, exist_ok=True)
os.makedirs(TESS_ORIGINAL_FOLDER_PATH, exist_ok=True)
