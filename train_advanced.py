"""
Advanced Training Launcher
Run this for maximum model performance (75-80%+ accuracy)

Usage:
    python train_advanced.py
"""

from src.model.advanced_trainer import main

if __name__ == "__main__":
    print("""
==================================================================
   ADVANCED EMOTION DETECTION TRAINING
   Portfolio-Grade Model Development
==================================================================

Features:
  - Face detection preprocessing
  - EfficientNetB2/B3 with attention mechanism
  - Advanced data augmentation
  - Class imbalance handling
  - Comprehensive monitoring
  
Target: 75-80%+ validation accuracy
Time: 2-4 hours (depends on hardware)
    """)
    
    print("\nStarting training...")
    print()
    
    main()

