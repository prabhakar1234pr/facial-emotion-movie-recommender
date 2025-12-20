# 🎭 Emotion Detection & Movie Recommendation System

A deep learning-powered web application that detects emotions from facial expressions and recommends movies based on your current mood.

## ✨ Features

- **Real-time Emotion Detection**: Uses a Convolutional Neural Network (CNN) to detect 7 different emotions
- **Confidence Scores**: Shows probability distribution across all emotion categories
- **TMDB Integration**: Rich movie data with posters, ratings, trailers, and descriptions
- **AI-Powered Insights**: Personalized movie explanations using Groq LLM
- **Beautiful UI**: Professional interface with movie posters and visual elements
- **Smart Recommendations**: Emotion-to-genre mapping for accurate suggestions
- **Modern Web Interface**: Built with Streamlit for a smooth user experience
- **Modular Architecture**: Clean, maintainable code structure following best practices

## 🎯 Supported Emotions

- 😠 **Anger**
- 🤢 **Disgust**
- 😨 **Fear**
- 😊 **Happy**
- 😢 **Sad**
- 😲 **Surprise**
- 😐 **Neutral**

## 🏗️ Project Structure

```
Emotion-Detection-Deep-Learning/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   └── emotion_model.py      # Model loading and prediction logic
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py      # Streamlit web interface
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_processing.py   # Image preprocessing utilities
│   │   └── movie_data.py         # Movie recommendation data
│   └── config/
│       ├── __init__.py
│       └── config.py             # Configuration settings
├── models/
│   └── emotion_detection_model.keras  # Trained CNN model
├── data/
│   └── movie_recommendations.json     # Movie database
├── images/
│   ├── train/                         # Training dataset
│   └── test/                          # Test dataset
├── tests/                             # Unit tests (coming soon)
├── app.py                             # Main application entry point
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (for capturing photos)
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prabhakar1234pr/facial-emotion-movie-recommender.git
   cd facial-emotion-movie-recommender
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**
   
   Create a `.env` file in the root directory:
   ```bash
   TMDB_API_KEY=your_tmdb_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   **Get your API keys:**
   - **TMDB**: Sign up at https://www.themoviedb.org/signup and get API key instantly
   - **Groq**: Sign up at https://console.groq.com/ for free LLM API access

5. **Verify model file exists**
   - Ensure `emotion_detection_model.keras` is in the `models/` directory
   - If not, you'll need to train the model first (see Training section)

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🎮 Usage

1. **Allow Camera Access**: Grant camera permissions when prompted by your browser
2. **Take a Photo**: Click the camera button to capture your facial expression
3. **View Results**: See your detected emotion with confidence scores and AI insights
4. **Explore Movies**: Browse personalized movie recommendations with:
   - High-quality posters
   - Ratings and release dates
   - Plot summaries
   - AI-powered explanations for why each movie matches your mood
   - Direct links to trailers

## 🧠 Model Architecture

**Production Model (Advanced):**

- **Input**: 48x48 RGB images (face-detected and cropped)
- **Base Model**: EfficientNetB2/B3 (pre-trained on ImageNet)
- **Custom Layers**:
  - Dual pooling (Global Average + Global Max)
  - Attention mechanism for feature focus
  - Dense layers with L2 regularization
  - Batch normalization and dropout
  - 7-class softmax output
- **Training Strategy**:
  - Transfer learning with fine-tuning
  - Face detection preprocessing
  - Advanced data augmentation (8 types)
  - Class-weighted loss for imbalance
  - Learning rate scheduling
  - Early stopping (patience=20)
  - Comprehensive monitoring

**Original Model (Baseline):**
- Simple CNN with 57% accuracy
- Included for comparison

### Model Performance

**Current Model:**
- **Validation Accuracy**: ~57% (needs improvement)
- **Dataset Size**: 28,000+ training images

**Advanced Model (Production-Grade):**
- **Target Accuracy**: 75-80%+
- **Method**: EfficientNetB2/B3 + Attention + Face Detection
- **Training**: Run `python train_advanced.py`
- **Features**: State-of-the-art architecture with comprehensive monitoring

## 📈 Advanced Model Training (Production-Grade)

The current model (~57% accuracy) has been significantly enhanced with state-of-the-art techniques:

### 🚀 Quick Training (Recommended)

```bash
# Run advanced training - Portfolio grade!
python train_advanced.py
```

### ✨ Advanced Features

**What makes this production-grade:**

1. **Face Detection Preprocessing** 🎯
   - Automatic face detection and cropping
   - Focuses training on facial features only
   - Removes background noise

2. **EfficientNetB2/B3 with Attention** 🧠
   - Transfer learning from ImageNet
   - Custom attention mechanism for feature focus
   - State-of-the-art architecture

3. **Advanced Data Augmentation** 🔄
   - Rotation, shifts, zoom, shear
   - Brightness and channel variations
   - Dramatically improves generalization

4. **Smart Class Balancing** ⚖️
   - Handles imbalanced dataset (436 vs 7,164 images)
   - Automatic class weight calculation
   - Fair learning across all emotions

5. **Comprehensive Monitoring** 📊
   - Real-time training visualization
   - Per-class accuracy metrics
   - JSON metrics export
   - Professional training plots

**Expected Results:** 75-80%+ accuracy ⭐

**Training Time:**
- With GPU: 2-3 hours
- With CPU: 6-8 hours

---

## 🛠️ Development

### Code Structure

The project follows a modular architecture with separation of concerns:

- **Model Layer** (`src/model/`): Handles model loading and predictions
- **UI Layer** (`src/ui/`): Streamlit interface and user interactions
- **Utils Layer** (`src/utils/`): Helper functions for image processing and data management
- **Config Layer** (`src/config/`): Centralized configuration management

### Key Classes

- `EmotionDetector`: Main model class for emotion prediction
- `EmotionApp`: Streamlit application controller
- `Config`: Configuration management

## 🔧 Configuration

**Environment Variables (.env):**
- `TMDB_API_KEY`: Your TMDB API key for movie data
- `GROQ_API_KEY`: Your Groq API key for AI explanations

**Application Settings (src/config/config.py):**
- Model path and parameters
- Emotion labels
- Image processing settings
- UI settings

## 📊 Dataset

The model is trained on facial expression images organized by emotion:

```
images/
├── train/
│   ├── angry/    (3,993 images)
│   ├── disgust/  (436 images)
│   ├── fear/     (4,103 images)
│   ├── happy/    (7,164 images)
│   ├── neutral/  (4,982 images)
│   ├── sad/      (4,938 images)
│   └── surprise/ (3,205 images)
└── test/
    └── (similar structure)
```

## 🚧 Future Enhancements

- [ ] Improve model accuracy with transfer learning
- [x] Integrate TMDb API for rich movie metadata ✅
- [x] Add AI-powered personalized recommendations ✅
- [ ] Add user authentication and preference tracking
- [ ] Implement emotion history tracking
- [ ] Real-time video emotion detection
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Docker deployment
- [ ] Deploy to Hugging Face Spaces

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Prabhakar Elavala**

## 🙏 Acknowledgments

- FER-2013 dataset creators
- TensorFlow and Keras teams
- Streamlit framework
- Movie data curators

---

**Note**: This project is designed for educational and portfolio purposes. For production use, additional security, privacy, and performance considerations should be addressed.

