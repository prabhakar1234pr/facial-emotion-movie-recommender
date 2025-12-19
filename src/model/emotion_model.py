"""
Emotion Detection Model Module
Handles model loading and prediction
"""

import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import logging

from ..config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Emotion Detector class for loading and running predictions
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion detector
        
        Args:
            model_path (str, optional): Path to the trained model. 
                                       If None, uses default from config.
        """
        self.config = Config()
        self.model_path = model_path or self.config.model_path
        self.model = None
        self.emotion_labels = self.config.emotion_labels
        self._load_model()
        
    def _load_model(self):
        """Load the trained Keras model"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.model = load_model(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, preprocessed_image):
        """
        Predict emotion from preprocessed image
        
        Args:
            preprocessed_image (np.ndarray): Preprocessed image array 
                                            Shape: (1, 48, 48, 1)
        
        Returns:
            tuple: (predicted_emotion, confidence_scores)
                - predicted_emotion (str): The predicted emotion label
                - confidence_scores (dict): Dictionary of emotion: confidence pairs
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Make prediction
        predictions = self.model.predict(preprocessed_image, verbose=0)
        
        # Get the index of highest probability
        max_index = np.argmax(predictions[0])
        predicted_emotion = self.emotion_labels[max_index]
        
        # Create confidence scores dictionary
        confidence_scores = {
            emotion: float(predictions[0][i]) 
            for i, emotion in enumerate(self.emotion_labels)
        }
        
        return predicted_emotion, confidence_scores
    
    def predict_from_array(self, image_array):
        """
        Predict emotion directly from a numpy array (for convenience)
        
        Args:
            image_array (np.ndarray): Image array (preprocessed or raw)
        
        Returns:
            tuple: (predicted_emotion, confidence_scores)
        """
        # Ensure proper shape
        if len(image_array.shape) == 2:
            # Add batch and channel dimensions
            image_array = image_array.reshape(1, *image_array.shape, 1)
        elif len(image_array.shape) == 3:
            # Add batch dimension
            image_array = image_array.reshape(1, *image_array.shape)
            
        return self.predict(image_array)

