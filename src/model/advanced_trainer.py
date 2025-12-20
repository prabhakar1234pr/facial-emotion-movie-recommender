"""
Advanced Model Training System - Production Grade
Target: 75-80%+ validation accuracy

Features:
- Face detection preprocessing
- EfficientNetB2/B3 with attention mechanism
- Advanced data augmentation
- Class imbalance handling
- Comprehensive monitoring and visualization
- Model ensemble capability
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
    Input, Multiply, Add, Concatenate, GlobalMaxPooling2D
)
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, LearningRateScheduler
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from datetime import datetime
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.config import Config


class FaceDetector:
    """Advanced face detection and preprocessing"""
    
    def __init__(self):
        """Initialize face detector"""
        # Try multiple cascade classifiers for better detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        print("[OK] Face detector initialized")
    
    def detect_and_crop(self, image, target_size=48):
        """
        Detect face and crop to focus on facial features
        
        Args:
            image (np.ndarray): Input grayscale image
            target_size (int): Output size
        
        Returns:
            np.ndarray: Cropped face or resized original
        """
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # Get largest face
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            x, y, w, h = faces[largest_idx]
            
            # Add padding around face (10%)
            padding = int(w * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop face
            face = image[y:y+h, x:x+w]
            
            # Resize
            return cv2.resize(face, (target_size, target_size))
        
        # Fallback: return resized original
        return cv2.resize(image, (target_size, target_size))
    
    def preprocess_batch(self, images, target_size=48):
        """
        Preprocess batch of images with face detection
        
        Args:
            images (list): List of images
            target_size (int): Output size
        
        Returns:
            np.ndarray: Preprocessed images
        """
        processed = []
        for img in images:
            processed.append(self.detect_and_crop(img, target_size))
        return np.array(processed)


class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention mechanism for better feature focus"""
    
    def __init__(self, units=512, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Attention mechanism
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait)
        
        # Apply attention
        return x * tf.expand_dims(a, -1)


class AdvancedEmotionTrainer:
    """Production-grade emotion detection trainer"""
    
    def __init__(self, train_folder_path=None, model_type='efficientnetb2'):
        """
        Initialize advanced trainer
        
        Args:
            train_folder_path (str, optional): Path to training data
            model_type (str): 'efficientnetb2' or 'efficientnetb3'
        """
        self.config = Config()
        self.train_folder_path = train_folder_path or str(
            Path(__file__).parent.parent.parent / "images" / "train"
        )
        self.emotions = self.config.emotion_labels
        self.model_type = model_type
        self.model = None
        self.class_weights = None
        self.face_detector = FaceDetector()
        self.history = None
        
        print("=" * 80)
        print(" ADVANCED EMOTION DETECTION TRAINING SYSTEM")
        print("=" * 80)
        print(f"Model: {model_type.upper()}")
        print("Features:")
        print("  [OK] Face detection preprocessing")
        print("  [OK] Transfer learning with attention")
        print("  [OK] Advanced data augmentation")
        print("  [OK] Class imbalance handling")
        print("  [OK] Learning rate scheduling")
        print("  [OK] Comprehensive monitoring")
        print(f"Target Accuracy: 75-80%+")
        print("=" * 80)
        
    def load_and_preprocess_images(self):
        """
        Load images with face detection preprocessing
        
        Returns:
            tuple: (images, labels) numpy arrays
        """
        print("\n" + "=" * 80)
        print("STEP 1: LOADING & PREPROCESSING DATA")
        print("=" * 80)
        
        images = []
        labels = []
        face_detected_count = 0
        
        # Map emotion names to folder names (handle both 'anger' and 'angry')
        emotion_folder_map = {
            'anger': 'angry',
            'disgust': 'disgust', 
            'fear': 'fear',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        for emotion in self.emotions:
            # Use mapped folder name
            folder_name = emotion_folder_map.get(emotion, emotion)
            emotion_folder = os.path.join(self.train_folder_path, folder_name)
            
            if not os.path.exists(emotion_folder):
                print(f"[WARNING]  Folder not found: {emotion_folder}")
                continue
            
            img_files = [f for f in os.listdir(emotion_folder) 
                        if f.endswith(('jpg', 'jpeg', 'png', 'gif'))]
            
            print(f"\n Processing {emotion}: {len(img_files)} images")
            
            for idx, img_name in enumerate(img_files):
                if idx % 500 == 0:
                    print(f"  Progress: {idx}/{len(img_files)}", end='\r')
                
                img_path = os.path.join(emotion_folder, img_name)
                
                try:
                    # Read image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Detect and crop face
                    face = self.face_detector.detect_and_crop(img, target_size=48)
                    
                    # Check if face was detected (compare with original resize)
                    if not np.array_equal(face, cv2.resize(img, (48, 48))):
                        face_detected_count += 1
                    
                    images.append(face)
                    labels.append(emotion)
                    
                except Exception as e:
                    continue
            
            print(f"  Progress: {len(img_files)}/{len(img_files)} [OK]")
        
        print(f"\n Preprocessing Summary:")
        print(f"  Total images: {len(images)}")
        print(f"  Faces detected: {face_detected_count} ({face_detected_count/len(images)*100:.1f}%)")
        
        return np.array(images), np.array(labels)
    
    def preprocess_data(self, images, labels):
        """
        Advanced preprocessing with class balancing
        
        Args:
            images (np.ndarray): Raw images
            labels (np.ndarray): String labels
        
        Returns:
            tuple: Preprocessed train and validation sets
        """
        print("\n" + "=" * 80)
        print("STEP 2: DATA PREPROCESSING & AUGMENTATION SETUP")
        print("=" * 80)
        
        # Normalize
        images = images / 255.0
        
        # Convert to RGB for transfer learning
        images = np.stack([images] * 3, axis=-1)
        print("[OK] Converted to RGB format")
        
        # Encode labels
        label_to_index = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        numeric_labels = np.array([label_to_index[label] for label in labels])
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(numeric_labels),
            y=numeric_labels
        )
        self.class_weights = dict(enumerate(class_weights))
        
        print("\n  Class Distribution & Weights:")
        for emotion, weight in zip(self.emotions, class_weights):
            count = np.sum(numeric_labels == label_to_index[emotion])
            print(f"  {emotion:10s}: {count:5d} images | Weight: {weight:.3f}")
        
        # One-hot encode
        labels_categorical = to_categorical(numeric_labels, num_classes=len(self.emotions))
        
        # Stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels_categorical,
            test_size=0.15,  # Smaller validation set for more training data
            random_state=42,
            stratify=numeric_labels
        )
        
        print(f"\n Data Split:")
        print(f"  Training:   {len(X_train):6d} samples ({len(X_train)/len(images)*100:.1f}%)")
        print(f"  Validation: {len(X_val):6d} samples ({len(X_val)/len(images)*100:.1f}%)")
        
        return X_train, X_val, y_train, y_val
    
    def build_advanced_model(self):
        """
        Build state-of-the-art model with attention mechanism
        
        Returns:
            keras.Model: Advanced compiled model
        """
        print("\n" + "=" * 80)
        print("STEP 3: BUILDING ADVANCED MODEL ARCHITECTURE")
        print("=" * 80)
        
        # Input
        input_layer = Input(shape=(48, 48, 3), name='input')
        
        # Base model selection
        if self.model_type == 'efficientnetb3':
            print("  Loading EfficientNetB3 (Better accuracy, slower)")
            base_model = EfficientNetB3(
                include_top=False,
                weights='imagenet',
                input_tensor=input_layer
            )
        else:
            print("  Loading EfficientNetB2 (Good balance)")
            base_model = EfficientNetB2(
                include_top=False,
                weights='imagenet',
                input_tensor=input_layer
            )
        
        # Freeze initial layers
        base_model.trainable = False
        
        # Get base model output
        x = base_model.output
        
        # Global pooling paths
        gap = GlobalAveragePooling2D()(x)
        gmp = GlobalMaxPooling2D()(x)
        
        # Combine pooling
        combined = Concatenate()([gap, gmp])
        
        # Get the combined features size dynamically
        combined_size = combined.shape[-1]
        
        # Attention mechanism - match the combined size
        attention = Dense(combined_size, activation='sigmoid', name='attention_gate')(combined)
        x = Multiply()([combined, attention])
        
        # Dense layers with regularization
        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output
        outputs = Dense(len(self.emotions), activation='softmax', name='output')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=outputs, name='AdvancedEmotionNet')
        
        # Compile with custom settings
        initial_lr = 0.0001
        optimizer = Adam(learning_rate=initial_lr)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
        )
        
        # Model summary
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        
        print(f"\n Model Architecture:")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable:        {total_params - trainable_params:,}")
        print(f"  Initial learning rate: {initial_lr}")
        
        return model
    
    def get_advanced_augmentation(self):
        """
        State-of-the-art data augmentation
        
        Returns:
            ImageDataGenerator: Advanced augmentation
        """
        return ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            shear_range=0.2,
            channel_shift_range=20.0,
            fill_mode='nearest'
        )
    
    def learning_rate_schedule(self, epoch, lr):
        """Custom learning rate schedule"""
        if epoch < 10:
            return lr
        elif epoch < 30:
            return lr * 0.9
        elif epoch < 50:
            return lr * 0.8
        else:
            return lr * 0.7
    
    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
        """
        Advanced training with all optimizations
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Maximum epochs
            batch_size (int): Batch size
        
        Returns:
            keras.callbacks.History: Training history
        """
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING MODEL")
        print("=" * 80)
        print(f"  Max epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples per epoch: {len(X_train)}")
        print(f"  Steps per epoch: {len(X_train) // batch_size}")
        print("=" * 80)
        
        # Build model
        self.model = self.build_advanced_model()
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = Path(self.config.model_path).parent
        
        callbacks = [
            ModelCheckpoint(
                str(models_dir / 'best_emotion_model.keras'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            LearningRateScheduler(self.learning_rate_schedule, verbose=0)
        ]
        
        # Data augmentation
        datagen = self.get_advanced_augmentation()
        datagen.fit(X_train)
        
        print("\n Training started!")
        print("=" * 80)
        
        # Train
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        print("\n" + "=" * 80)
        print("[OK] Training completed!")
        print("=" * 80)
        
        return self.history
    
    def evaluate(self, X_val, y_val):
        """
        Comprehensive model evaluation
        
        Args:
            X_val, y_val: Validation data
        
        Returns:
            dict: Detailed metrics
        """
        print("\n" + "=" * 80)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 80)
        
        # Overall metrics
        results = self.model.evaluate(X_val, y_val, verbose=0)
        val_loss, val_accuracy, top2_accuracy = results
        
        # Per-class accuracy
        predictions = self.model.predict(X_val, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_val, axis=1)
        
        print("\n OVERALL PERFORMANCE:")
        print("=" * 80)
        print(f"  Loss:         {val_loss:.4f}")
        print(f"  Accuracy:     {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"  Top-2 Acc:    {top2_accuracy:.4f} ({top2_accuracy*100:.2f}%)")
        
        print("\n PER-CLASS ACCURACY:")
        print("=" * 80)
        for idx, emotion in enumerate(self.emotions):
            mask = true_classes == idx
            if mask.sum() > 0:
                class_acc = (pred_classes[mask] == idx).mean()
                count = mask.sum()
                print(f"  {emotion:10s}: {class_acc*100:5.2f}% ({count:4d} samples)")
        
        # Performance assessment
        print("\n ASSESSMENT:")
        print("=" * 80)
        if val_accuracy >= 0.80:
            print("   OUTSTANDING! Ready for production deployment!")
        elif val_accuracy >= 0.75:
            print("  [OK] EXCELLENT! Exceeds target, ready for deployment!")
        elif val_accuracy >= 0.70:
            print("  [OK] GOOD! Meets target accuracy for deployment!")
        elif val_accuracy >= 0.65:
            print("  [WARNING]  ACCEPTABLE but consider longer training")
        else:
            print("  [WARNING]  NEEDS IMPROVEMENT - Check data quality")
        
        print("=" * 80)
        
        return {
            'loss': val_loss,
            'accuracy': val_accuracy,
            'top2_accuracy': top2_accuracy
        }
    
    def save_training_report(self):
        """Generate comprehensive training report"""
        print("\n" + "=" * 80)
        print("STEP 6: GENERATING TRAINING REPORT")
        print("=" * 80)
        
        # Save history plot
        self._plot_training_history()
        
        # Save metrics to JSON
        metrics = {
            'model_type': self.model_type,
            'final_accuracy': float(max(self.history.history['val_accuracy'])),
            'final_loss': float(min(self.history.history['val_loss'])),
            'epochs_trained': len(self.history.history['loss']),
            'best_epoch': int(np.argmax(self.history.history['val_accuracy'])) + 1,
            'emotions': self.emotions,
            'class_weights': {k: float(v) for k, v in self.class_weights.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open('training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("[OK] Saved training_history.png")
        print("[OK] Saved training_metrics.json")
        print("=" * 80)
    
    def _plot_training_history(self):
        """Create detailed training visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced Training Report', fontsize=16, fontweight='bold')
        
        # Accuracy
        ax = axes[0, 0]
        ax.plot(self.history.history['accuracy'], label='Training', linewidth=2)
        ax.plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax.set_title('Model Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss
        ax = axes[0, 1]
        ax.plot(self.history.history['loss'], label='Training', linewidth=2)
        ax.plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        ax.set_title('Model Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top-2 Accuracy
        ax = axes[1, 0]
        ax.plot(self.history.history['top2_accuracy'], label='Training', linewidth=2)
        ax.plot(self.history.history['val_top2_accuracy'], label='Validation', linewidth=2)
        ax.set_title('Top-2 Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Top-2 Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        ax = axes[1, 1]
        if 'lr' in self.history.history:
            ax.plot(self.history.history['lr'], linewidth=2, color='orange')
            ax.set_title('Learning Rate Schedule', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            # Summary statistics
            ax.axis('off')
            best_acc = max(self.history.history['val_accuracy'])
            best_epoch = np.argmax(self.history.history['val_accuracy']) + 1
            final_acc = self.history.history['val_accuracy'][-1]
            
            summary = f"""
            TRAINING SUMMARY
            
            Best Validation Accuracy: {best_acc*100:.2f}%
            Best Epoch: {best_epoch}
            Final Accuracy: {final_acc*100:.2f}%
            Total Epochs: {len(self.history.history['loss'])}
            
            Model: {self.model_type.upper()}
            """
            ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
                   verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main advanced training pipeline"""
    print("\n" + "=" * 80)
    print("ADVANCED EMOTION DETECTION TRAINING")
    print("Portfolio & Resume Grade System")
    print("=" * 80)
    
    # Model selection - using default for automated training
    print("\nUsing EfficientNetB2 (Recommended - Good balance)")
    print("For EfficientNetB3, edit advanced_trainer.py and change model_type")
    
    model_type = 'efficientnetb2'  # Default to B2 for good balance
    
    # Initialize trainer
    trainer = AdvancedEmotionTrainer(model_type=model_type)
    
    # Load and preprocess
    images, labels = trainer.load_and_preprocess_images()
    
    if len(images) == 0:
        raise ValueError("[ERROR] No images found! Check your data path.")
    
    # Preprocess
    X_train, X_val, y_train, y_val = trainer.preprocess_data(images, labels)
    
    # Train
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=150, batch_size=32)
    
    # Evaluate
    metrics = trainer.evaluate(X_val, y_val)
    
    # Generate report
    trainer.save_training_report()
    
    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n[OK] Best model saved: models/best_emotion_model.keras")
    print(f"[OK] Training plots: training_history.png")
    print(f"[OK] Metrics report: training_metrics.json")
    print(f"\n Final Accuracy: {metrics['accuracy']*100:.2f}%")
    
    if metrics['accuracy'] >= 0.75:
        print("\n YOUR MODEL IS READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("1. Test locally: streamlit run app.py")
        print("2. Deploy to Hugging Face Spaces")
        print("3. Add to your resume/portfolio")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

