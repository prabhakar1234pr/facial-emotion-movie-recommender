"""
Model Training Script
Contains code to train the emotion detection model
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.config import Config


class EmotionModelTrainer:
    """Class to handle model training"""
    
    def __init__(self, train_folder_path=None):
        """
        Initialize the trainer
        
        Args:
            train_folder_path (str, optional): Path to training data
        """
        self.config = Config()
        self.train_folder_path = train_folder_path or str(self.config.TRAIN_DATA_PATH)
        self.emotions = self.config.emotion_labels
        self.model = None
        
    def load_images(self):
        """
        Load and preprocess training images
        
        Returns:
            tuple: (images, labels) numpy arrays
        """
        images = []
        labels = []
        
        for emotion in self.emotions:
            emotion_folder = os.path.join(self.train_folder_path, emotion)
            
            if not os.path.exists(emotion_folder):
                print(f"Folder not found: {emotion_folder}")
                continue
            
            print(f"Loading {emotion} images...")
            for img_name in os.listdir(emotion_folder):
                img_path = os.path.join(emotion_folder, img_name)
                
                if not img_name.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    continue
                
                # Read in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Resize to 48x48 pixels
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(emotion)
        
        return np.array(images), np.array(labels)
    
    def preprocess_data(self, images, labels):
        """
        Preprocess images and labels
        
        Args:
            images (np.ndarray): Raw images
            labels (np.ndarray): String labels
        
        Returns:
            tuple: Preprocessed train and validation sets
        """
        # Normalize pixel values
        images = images / 255.0
        
        # Encode labels
        label_to_index = {emotion: index for index, emotion in enumerate(self.emotions)}
        labels = np.array([label_to_index[label] for label in labels])
        labels = to_categorical(labels, num_classes=len(self.emotions))
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        
        # Reshape for CNN input
        X_train = X_train.reshape(-1, 48, 48, 1)
        X_val = X_val.reshape(-1, 48, 48, 1)
        
        return X_train, X_val, y_train, y_val
    
    def build_model(self):
        """
        Build the CNN model architecture
        
        Returns:
            keras.Model: Compiled model
        """
        model = Sequential([
            # First convolutional layer
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional layer
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third convolutional layer
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        
        Returns:
            keras.callbacks.History: Training history
        """
        # Build model
        self.model = self.build_model()
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            str(self.config.model_path),
            save_best_only=True
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        datagen.fit(X_train)
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        return history
    
    def evaluate(self, X_val, y_val):
        """
        Evaluate the model
        
        Args:
            X_val, y_val: Validation data
        
        Returns:
            tuple: (loss, accuracy)
        """
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val)
        print(f'\nValidation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        return val_loss, val_accuracy


def main():
    """Main training script"""
    print("Starting model training...")
    
    # Initialize trainer
    trainer = EmotionModelTrainer()
    
    # Load images
    print("\n1. Loading images...")
    images, labels = trainer.load_images()
    print(f"Loaded {len(images)} images")
    
    if len(images) == 0:
        raise ValueError("No images found! Check your data path.")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train, X_val, y_train, y_val = trainer.preprocess_data(images, labels)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train model
    print("\n3. Training model...")
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\n4. Evaluating model...")
    trainer.evaluate(X_val, y_val)
    
    print("\n✅ Training complete! Model saved to models/emotion_detection_model.keras")


if __name__ == "__main__":
    main()

