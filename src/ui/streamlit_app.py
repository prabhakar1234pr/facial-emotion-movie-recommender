"""
Streamlit Web Application for Emotion Detection
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.emotion_model import EmotionDetector
from src.utils.image_processing import preprocess_image, decode_image_from_bytes
from src.utils.movie_data import get_movie_recommendations
from src.config.config import Config


class EmotionApp:
    """Main application class for Streamlit UI"""
    
    def __init__(self):
        """Initialize the application"""
        self.config = Config()
        self.detector = None
        self._setup_page()
        
    def _setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=self.config.app_title,
            page_icon="🎭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def _load_detector(self):
        """Load the emotion detector model (cached)"""
        if self.detector is None:
            with st.spinner("Loading emotion detection model..."):
                try:
                    self.detector = EmotionDetector()
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.stop()
        return self.detector
    
    def _display_header(self):
        """Display application header"""
        st.title("🎭 " + self.config.app_title)
        st.markdown(
            """
            ### Discover movies that match your mood!
            
            This application uses deep learning to detect your emotion from a photo 
            and recommends movies that align with your current mood.
            """
        )
        st.divider()
    
    def _display_sidebar(self):
        """Display sidebar with information"""
        with st.sidebar:
            st.header("ℹ️ About")
            st.markdown(
                """
                **How it works:**
                1. 📸 Take a picture using your camera
                2. 🤖 AI detects your emotion
                3. 🎬 Get movie recommendations
                
                **Supported Emotions:**
                - 😠 Anger
                - 🤢 Disgust
                - 😨 Fear
                - 😊 Happy
                - 😢 Sad
                - 😲 Surprise
                - 😐 Neutral
                """
            )
            
            st.divider()
            
            st.markdown(
                """
                **Tips for best results:**
                - Ensure good lighting
                - Face the camera directly
                - Show clear facial expressions
                """
            )
    
    def _display_emotion_results(self, emotion, confidence_scores):
        """Display emotion detection results with confidence scores"""
        
        # Main emotion result
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Detected Emotion")
            # Emotion emoji mapping
            emoji_map = {
                'anger': '😠',
                'disgust': '🤢',
                'fear': '😨',
                'happy': '😊',
                'sad': '😢',
                'surprise': '😲',
                'neutral': '😐'
            }
            emoji = emoji_map.get(emotion, '🎭')
            st.markdown(f"# {emoji} **{emotion.upper()}**")
            st.metric("Confidence", f"{confidence_scores[emotion]*100:.1f}%")
        
        with col2:
            st.subheader("All Emotion Probabilities")
            # Sort by confidence
            sorted_emotions = sorted(
                confidence_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Display as progress bars
            for emo, conf in sorted_emotions:
                st.write(f"**{emo.capitalize()}**")
                st.progress(conf)
                st.caption(f"{conf*100:.2f}%")
    
    def _display_movie_recommendations(self, emotion):
        """Display movie recommendations based on emotion"""
        st.divider()
        st.subheader("🎬 Movie Recommendations for You")
        
        # Get multiple recommendations
        movies = get_movie_recommendations(emotion, count=5)
        
        st.markdown(
            f"""
            Based on your **{emotion}** emotion, we recommend these movies:
            """
        )
        
        # Display movies in columns
        cols = st.columns(5)
        for idx, movie in enumerate(movies):
            with cols[idx]:
                st.markdown(f"**{idx+1}. {movie}**")
        
        # Show all available movies for this emotion
        with st.expander("View all recommendations for this emotion"):
            all_movies = get_movie_recommendations(emotion, count=100)
            for i, movie in enumerate(all_movies, 1):
                st.write(f"{i}. {movie}")
    
    def run(self):
        """Main application loop"""
        self._display_header()
        self._display_sidebar()
        
        # Load detector
        detector = self._load_detector()
        
        # Main content area
        st.subheader("📸 Capture Your Emotion")
        
        # Camera input
        img_file = st.camera_input("Take a picture")
        
        if img_file:
            # Read image bytes
            file_bytes = img_file.read()
            
            # Decode image
            img = decode_image_from_bytes(file_bytes)
            
            # Create columns for image and results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(img, channels="BGR", caption="Your Photo", use_container_width=True)
            
            with col2:
                with st.spinner("Analyzing your emotion..."):
                    # Preprocess image
                    preprocessed_img = preprocess_image(img)
                    
                    # Predict emotion
                    emotion, confidence_scores = detector.predict(preprocessed_img)
            
            # Display results
            st.divider()
            self._display_emotion_results(emotion, confidence_scores)
            
            # Display movie recommendations
            self._display_movie_recommendations(emotion)
            
        else:
            st.info("👆 Click the camera button above to take a picture!")
            
            # Show demo info
            st.markdown(
                """
                ---
                ### 🎯 What you'll get:
                - Real-time emotion detection
                - Confidence scores for all emotions
                - Personalized movie recommendations
                - Multiple movie suggestions to choose from
                """
            )


def main():
    """Main entry point for Streamlit app"""
    app = EmotionApp()
    app.run()


if __name__ == "__main__":
    main()

