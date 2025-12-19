"""
Improved Model Training Script with Transfer Learning
Target: 70-75%+ validation accuracy
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
    Conv2D, MaxPooling2D, Flatten, Input
)
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.config import Config


class ImprovedEmotionTrainer:
    """Enhanced trainer with transfer learning and advanced techniques"""
    
    def __init__(self, train_folder_path=None, use_transfer_learning=True):
        """
        Initialize the improved trainer
        
        Args:
            train_folder_path (str, optional): Path to training data
            use_transfer_learning (bool): Use pre-trained model (recommended)
        """
        self.config = Config()
        self.train_folder_path = train_folder_path or str(Path(__file__).parent.parent.parent / "images" / "train")
        self.emotions = self.config.emotion_labels
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        self.class_weights = None
        
        print(f"🚀 Improved Emotion Trainer")
        print(f"Transfer Learning: {use_transfer_learning}")
        print(f"Target Accuracy: 70%+")
        
    def load_images_efficiently(self):
        """
        Load and preprocess training images with progress tracking
        
        Returns:
            tuple: (images, labels) numpy arrays
        """
        images = []
        labels = []
        
        print("\n📂 Loading images from each emotion category...")
        
        for emotion in self.emotions:
            emotion_folder = os.path.join(self.train_folder_path, emotion)
            
            if not os.path.exists(emotion_folder):
                print(f"⚠️  Folder not found: {emotion_folder}")
                continue
            
            img_files = [f for f in os.listdir(emotion_folder) 
                        if f.endswith(('jpg', 'jpeg', 'png', 'gif'))]
            
            print(f"  {emotion}: {len(img_files)} images")
            
            for img_name in img_files:
                img_path = os.path.join(emotion_folder, img_name)
                
                try:
                    # Read in grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    # Resize to 48x48 pixels
                    img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(emotion)
                    
                except Exception as e:
                    continue
        
        print(f"\n✅ Total images loaded: {len(images)}")
        return np.array(images), np.array(labels)
    
    def preprocess_data(self, images, labels):
        """
        Preprocess images and labels with class imbalance handling
        
        Args:
            images (np.ndarray): Raw images
            labels (np.ndarray): String labels
        
        Returns:
            tuple: Preprocessed train and validation sets with class weights
        """
        print("\n🔧 Preprocessing data...")
        
        # Normalize pixel values
        images = images / 255.0
        
        # For transfer learning, convert to RGB
        if self.use_transfer_learning:
            images = np.stack([images] * 3, axis=-1)
            print("  Converted to RGB for transfer learning")
        else:
            images = images.reshape(-1, 48, 48, 1)
            print("  Using grayscale images")
        
        # Encode labels
        label_to_index = {emotion: index for index, emotion in enumerate(self.emotions)}
        numeric_labels = np.array([label_to_index[label] for label in labels])
        
        # Calculate class weights to handle imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(numeric_labels),
            y=numeric_labels
        )
        self.class_weights = dict(enumerate(class_weights))
        
        print("\n⚖️  Class weights (to handle imbalance):")
        for emotion, weight in zip(self.emotions, class_weights):
            print(f"  {emotion}: {weight:.2f}")
        
        # One-hot encode labels
        labels_categorical = to_categorical(numeric_labels, num_classes=len(self.emotions))
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels_categorical, 
            test_size=0.2, 
            random_state=42,
            stratify=numeric_labels
        )
        
        print(f"\n📊 Data split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def build_transfer_learning_model(self):
        """
        Build model with transfer learning (EfficientNetB0)
        Better accuracy than training from scratch
        
        Returns:
            keras.Model: Compiled model
        """
        print("\n🏗️  Building transfer learning model (EfficientNetB0)...")
        
        # Input layer
        input_layer = Input(shape=(48, 48, 3))
        
        # Load pre-trained EfficientNetB0 (without top layers)
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=input_layer,
            pooling='avg'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom top layers
        x = base_model.output
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(len(self.emotions), activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=outputs)
        
        # Compile with lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"  Total parameters: {model.count_params():,}")
        print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def build_custom_cnn_model(self):
        """
        Build improved custom CNN (better than original)
        
        Returns:
            keras.Model: Compiled model
        """
        print("\n🏗️  Building improved custom CNN...")
        
        if self.use_transfer_learning:
            input_shape = (48, 48, 3)
        else:
            input_shape = (48, 48, 1)
        
        model = Sequential([
            # Block 1
            Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 4
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(len(self.emotions), activation='softmax')
        ])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def get_data_augmentation(self):
        """
        Enhanced data augmentation for better generalization
        
        Returns:
            ImageDataGenerator: Augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the model with all improvements
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size
        
        Returns:
            keras.callbacks.History: Training history
        """
        print(f"\n🎯 Starting training...")
        print(f"  Max epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        
        # Build model
        if self.use_transfer_learning:
            self.model = self.build_transfer_learning_model()
        else:
            self.model = self.build_custom_cnn_model()
        
        # Setup callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(Path(self.config.model_path).parent / 'best_emotion_model.keras'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Data augmentation
        datagen = self.get_data_augmentation()
        datagen.fit(X_train)
        
        print("\n🚀 Training started!")
        print("=" * 60)
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        print("\n" + "=" * 60)
        print("✅ Training completed!")
        
        return history
    
    def evaluate(self, X_val, y_val):
        """
        Evaluate the model and show detailed metrics
        
        Args:
            X_val, y_val: Validation data
        
        Returns:
            tuple: (loss, accuracy)
        """
        print("\n📊 Evaluating model...")
        
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f'\n{"="*60}')
        print(f'🎯 FINAL RESULTS:')
        print(f'{"="*60}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)')
        print(f'{"="*60}')
        
        if val_accuracy >= 0.70:
            print("🎉 EXCELLENT! Model achieved 70%+ accuracy!")
        elif val_accuracy >= 0.65:
            print("✅ GOOD! Model shows significant improvement!")
        else:
            print("⚠️  Consider training longer or adjusting hyperparameters")
        
        return val_loss, val_accuracy
    
    def plot_training_history(self, history):
        """Plot training history"""
        try:
            plt.figure(figsize=(12, 4))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
            print("\n📈 Training plots saved to: training_history.png")
            
        except Exception as e:
            print(f"\n⚠️  Could not create plots: {e}")


def main():
    """Main training script"""
    print("=" * 60)
    print("🚀 IMPROVED EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)
    print("\nTarget: 70-75%+ accuracy (significant improvement over 57%)")
    print("\nFeatures:")
    print("  ✅ Transfer Learning (EfficientNetB0)")
    print("  ✅ Class imbalance handling")
    print("  ✅ Advanced data augmentation")
    print("  ✅ Learning rate scheduling")
    print("  ✅ Early stopping")
    print("=" * 60)
    
    # Ask user for preference
    print("\nChoose training method:")
    print("1. Transfer Learning (Recommended - Faster, Better accuracy)")
    print("2. Custom CNN (Train from scratch)")
    
    choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
    use_transfer = choice == "1"
    
    # Initialize trainer
    trainer = ImprovedEmotionTrainer(use_transfer_learning=use_transfer)
    
    # Load images
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    images, labels = trainer.load_images_efficiently()
    
    if len(images) == 0:
        raise ValueError("❌ No images found! Check your data path.")
    
    # Preprocess data
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)
    X_train, X_val, y_train, y_val = trainer.preprocess_data(images, labels)
    
    # Train model
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING MODEL")
    print("=" * 60)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATION")
    print("=" * 60)
    trainer.evaluate(X_val, y_val)
    
    # Plot history
    trainer.plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n✅ Best model saved to: models/best_emotion_model.keras")
    print("✅ Training plots saved to: training_history.png")
    print("\nNext steps:")
    print("1. Test the model with sample images")
    print("2. Deploy to Hugging Face Spaces")
    print("3. Update app.py to use the new model")


if __name__ == "__main__":
    main()

