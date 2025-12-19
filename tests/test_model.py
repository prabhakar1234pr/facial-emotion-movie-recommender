"""
Unit tests for emotion detection model
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.emotion_model import EmotionDetector
from src.utils.image_processing import preprocess_image


class TestEmotionDetector(unittest.TestCase):
    """Test cases for EmotionDetector class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Create a dummy preprocessed image
        cls.dummy_image = np.random.rand(1, 48, 48, 1).astype(np.float32)
    
    def test_model_loads(self):
        """Test that model loads without errors"""
        try:
            detector = EmotionDetector()
            self.assertIsNotNone(detector.model)
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    def test_predict_shape(self):
        """Test that prediction returns correct format"""
        detector = EmotionDetector()
        emotion, confidence_scores = detector.predict(self.dummy_image)
        
        # Check emotion is string
        self.assertIsInstance(emotion, str)
        
        # Check confidence_scores is dict
        self.assertIsInstance(confidence_scores, dict)
        
        # Check all emotions have scores
        self.assertEqual(len(confidence_scores), 7)
    
    def test_emotion_labels(self):
        """Test that emotion labels are correct"""
        detector = EmotionDetector()
        expected_emotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.assertEqual(detector.emotion_labels, expected_emotions)


class TestImageProcessing(unittest.TestCase):
    """Test cases for image processing utilities"""
    
    def test_preprocess_image(self):
        """Test image preprocessing"""
        # Create a dummy BGR image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Preprocess
        processed = preprocess_image(dummy_image)
        
        # Check shape
        self.assertEqual(processed.shape, (1, 48, 48, 1))
        
        # Check normalization (values should be between 0 and 1)
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))


if __name__ == '__main__':
    unittest.main()

