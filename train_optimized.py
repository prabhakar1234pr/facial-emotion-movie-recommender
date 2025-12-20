"""
Optimized Emotion Detection Training
Target: 65-70% validation accuracy in 3-4 hours

This version:
- NO face detection (was causing problems)
- Proven EfficientNetB0 architecture
- Optimal data augmentation
- Proper learning rate and early stopping
- Class imbalance handling
- Reliable and tested approach
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent))
from src.config.config import Config


def load_images(train_folder_path, emotions):
    """Load all training images"""
    print("\n" + "="*60)
    print("STEP 1: LOADING IMAGES")
    print("="*60)
    
    # Map emotion labels to folder names
    emotion_folders = {
        'anger': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'sad': 'sad',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }
    
    images = []
    labels = []
    
    for emotion in emotions:
        folder_name = emotion_folders.get(emotion, emotion)
        emotion_folder = os.path.join(train_folder_path, folder_name)
        
        if not os.path.exists(emotion_folder):
            print(f"[WARNING] Folder not found: {emotion_folder}")
            continue
        
        img_files = [f for f in os.listdir(emotion_folder) 
                    if f.endswith(('jpg', 'jpeg', 'png'))]
        
        print(f"Loading {emotion}: {len(img_files)} images")
        
        for idx, img_name in enumerate(img_files):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{len(img_files)}", end='\r')
            
            img_path = os.path.join(emotion_folder, img_name)
            
            try:
                # Simple resize - NO face detection
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(emotion)
                
            except Exception:
                continue
        
        print(f"  Loaded: {len([l for l in labels if l == emotion])} images")
    
    print(f"\n[OK] Total images loaded: {len(images)}")
    return np.array(images), np.array(labels)


def preprocess_data(images, labels, emotions):
    """Preprocess and split data"""
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING DATA")
    print("="*60)
    
    # Normalize
    images = images / 255.0
    
    # Convert to RGB (for transfer learning)
    images = np.stack([images] * 3, axis=-1)
    
    # Encode labels
    label_to_index = {emotion: idx for idx, emotion in enumerate(emotions)}
    numeric_labels = np.array([label_to_index[label] for label in labels])
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(numeric_labels),
        y=numeric_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    print("\nClass distribution:")
    for emotion, weight in zip(emotions, class_weights):
        count = np.sum(numeric_labels == label_to_index[emotion])
        print(f"  {emotion:10s}: {count:5d} images | Weight: {weight:.3f}")
    
    # One-hot encode
    labels_categorical = to_categorical(numeric_labels, num_classes=len(emotions))
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_categorical,
        test_size=0.2,
        random_state=42,
        stratify=numeric_labels
    )
    
    print(f"\nData split:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    
    return X_train, X_val, y_train, y_val, class_weights_dict


def build_model(num_classes):
    """Build optimized model"""
    print("\n" + "="*60)
    print("STEP 3: BUILDING MODEL")
    print("="*60)
    print("Using EfficientNetB0 (proven architecture)")
    
    # Input
    input_layer = Input(shape=(48, 48, 3))
    
    # Base model
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer,
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Custom top layers
    x = base_model.output
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=input_layer, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"[OK] Model built: {model.count_params():,} parameters")
    return model


def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    """Train with optimal settings"""
    print("\n" + "="*60)
    print("STEP 4: TRAINING MODEL")
    print("="*60)
    print("Expected time: 3-4 hours")
    print("Target accuracy: 65-70%")
    print("="*60)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # More patience than before
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            'models/best_emotion_model.keras',
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
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    datagen.fit(X_train)
    
    print("\n[START] Training beginning...")
    print("="*60)
    
    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("[OK] Training completed!")
    print("="*60)
    
    return history


def evaluate_and_save(model, X_val, y_val, history, emotions):
    """Final evaluation and save results"""
    print("\n" + "="*60)
    print("STEP 5: FINAL EVALUATION")
    print("="*60)
    
    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nFINAL RESULTS:")
    print(f"  Validation Loss:     {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Per-class accuracy
    predictions = model.predict(X_val, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_val, axis=1)
    
    print(f"\nPer-class accuracy:")
    for idx, emotion in enumerate(emotions):
        mask = true_classes == idx
        if mask.sum() > 0:
            class_acc = (pred_classes[mask] == idx).mean()
            print(f"  {emotion:10s}: {class_acc*100:5.2f}%")
    
    # Assessment
    print(f"\nASSESSMENT:")
    if val_accuracy >= 0.70:
        print("  [EXCELLENT] Target exceeded! Ready for deployment!")
    elif val_accuracy >= 0.65:
        print("  [GOOD] Target met! Model is production-ready!")
    elif val_accuracy >= 0.60:
        print("  [ACCEPTABLE] Close to target, usable for portfolio")
    else:
        print("  [NEEDS WORK] Consider training longer")
    
    # Save training plots
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\n[OK] Training plots saved: training_history.png")
    
    return val_accuracy


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("OPTIMIZED EMOTION DETECTION TRAINING")
    print("Target: 65-70% Accuracy | Time: 3-4 hours")
    print("="*60)
    
    config = Config()
    emotions = config.emotion_labels
    train_path = Path(__file__).parent / "images" / "train"
    
    # Load data
    images, labels = load_images(str(train_path), emotions)
    
    if len(images) == 0:
        print("[ERROR] No images found!")
        return
    
    # Preprocess
    X_train, X_val, y_train, y_val, class_weights = preprocess_data(
        images, labels, emotions
    )
    
    # Build model
    model = build_model(len(emotions))
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val, class_weights)
    
    # Evaluate
    final_accuracy = evaluate_and_save(model, X_val, y_val, history, emotions)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"[OK] Model saved: models/best_emotion_model.keras")
    print(f"[OK] Plots saved: training_history.png")
    print(f"\nFinal Accuracy: {final_accuracy*100:.2f}%")
    
    if final_accuracy >= 0.65:
        print("\n[SUCCESS] Model ready for deployment!")
        print("\nNext steps:")
        print("1. Test: streamlit run app.py")
        print("2. Deploy to Hugging Face Spaces")
        print("3. Add to your resume!")
    
    print("="*60)


if __name__ == "__main__":
    main()

