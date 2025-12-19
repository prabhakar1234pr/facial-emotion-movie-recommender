"""
Configuration settings for the application
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Model configuration
MODEL_PATH = BASE_DIR / "models" / "emotion_detection_model.keras"
MODEL_INPUT_SIZE = (48, 48)
MODEL_CHANNELS = 1  # Grayscale

# Emotion categories
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Image processing
IMAGE_SIZE = 48
NORMALIZATION_FACTOR = 255.0

# Data paths
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_PATH = BASE_DIR / "images" / "train"
TEST_DATA_PATH = BASE_DIR / "images" / "test"

# UI Configuration
APP_TITLE = "Emotion-Based Movie Recommendation System"
APP_DESCRIPTION = "Capture your emotion and get personalized movie recommendations"


class Config:
    """Configuration class for easy access to settings"""
    
    def __init__(self):
        self.model_path = str(MODEL_PATH)
        self.input_size = MODEL_INPUT_SIZE
        self.channels = MODEL_CHANNELS
        self.emotion_labels = EMOTION_LABELS
        self.image_size = IMAGE_SIZE
        self.normalization_factor = NORMALIZATION_FACTOR
        self.app_title = APP_TITLE
        self.app_description = APP_DESCRIPTION
        
    def validate_paths(self):
        """Validate that required paths exist"""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        return True

