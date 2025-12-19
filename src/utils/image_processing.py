"""
Image Processing Utilities
Handles all image preprocessing operations
"""

import cv2
import numpy as np
from pathlib import Path
import logging

from ..config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image, target_size=48):
    """
    Preprocess image for emotion detection
    
    Args:
        image (np.ndarray): Input image (BGR or grayscale)
        target_size (int): Target size for resizing (default: 48)
    
    Returns:
        np.ndarray: Preprocessed image ready for model input
                    Shape: (1, target_size, target_size, 1)
    """
    config = Config()
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize pixel values
    image = image / config.normalization_factor
    
    # Reshape for model input: (1, height, width, channels)
    image = np.reshape(image, (1, target_size, target_size, 1))
    
    return image


def load_and_preprocess_image(image_path, target_size=48):
    """
    Load an image from file and preprocess it
    
    Args:
        image_path (str or Path): Path to the image file
        target_size (int): Target size for resizing (default: 48)
    
    Returns:
        np.ndarray: Preprocessed image ready for model input
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image in grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    logger.info(f"Loaded image: {image_path.name}")
    
    return preprocess_image(image, target_size)


def decode_image_from_bytes(file_bytes):
    """
    Decode image from bytes (useful for Streamlit file uploads)
    
    Args:
        file_bytes (bytes): Image file bytes
    
    Returns:
        np.ndarray: Decoded image (BGR format)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image from bytes")
    
    return image


def apply_face_detection(image):
    """
    Detect and crop face from image (optional enhancement)
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        np.ndarray: Cropped face image or original if no face detected
    """
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    # If face detected, crop to face region
    if len(faces) > 0:
        # Get the first (largest) face
        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]
        logger.info("Face detected and cropped")
        return face_img
    
    logger.warning("No face detected, using full image")
    return image

