"""
Streamlit Web Application for Emotion Detection with Enhanced Movie Recommendations
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.emotion_model import EmotionDetector
from src.utils.image_processing import preprocess_image, decode_image_from_bytes
from src.utils.tmdb_api import TMDBClient
from src.utils.llm_recommender import LLMRecommender
from src.config.config import Config


class EmotionApp:
    """Main application class for Streamlit UI"""
    
    def __init__(self):
        """Initialize the application"""
        self.config = Config()
        self.detector = None
        self.tmdb_client = None
        self.llm_recommender = None
        self._setup_page()
        
    def _setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=self.config.app_title,
            page_icon="🎭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
            <style>
            .movie-card {
                border-radius: 10px;
                padding: 10px;
                background-color: #f0f2f6;
                margin: 10px 0;
            }
            .emotion-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                margin: 5px;
            }
            .ai-insight {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
            }
            </style>
        """, unsafe_allow_html=True)
        
    def _load_detector(self):
        """Load the emotion detector model (cached)"""
        if self.detector is None:
            with st.spinner("Loading emotion detection model..."):
                try:
                    self.detector = EmotionDetector()
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.stop()
        return self.detector
    
    def _load_tmdb_client(self):
        """Load TMDB client (cached)"""
        if self.tmdb_client is None:
            try:
                self.tmdb_client = TMDBClient()
            except Exception as e:
                st.warning(f"⚠️ TMDB API not available: {e}")
                self.tmdb_client = None
        return self.tmdb_client
    
    def _load_llm_recommender(self):
        """Load LLM recommender (cached)"""
        if self.llm_recommender is None:
            try:
                self.llm_recommender = LLMRecommender()
            except Exception as e:
                st.warning(f"⚠️ LLM API not available: {e}")
                self.llm_recommender = None
        return self.llm_recommender
    
    def _display_header(self):
        """Display application header"""
        st.title("🎭 " + self.config.app_title)
        st.markdown(
            """
            ### Discover movies that match your mood!
            
            This AI-powered application detects your emotion from a photo and recommends 
            movies with rich details, posters, and personalized explanations.
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
                3. 🎬 Get personalized movie recommendations
                4. 🎯 See ratings, posters, and AI insights
                
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
                **Features:**
                - 🎭 Deep Learning emotion detection
                - 🎬 TMDB movie database
                - 🤖 AI-powered explanations
                - ⭐ Ratings and reviews
                - 🎥 Movie trailers
                """
            )
            
            st.divider()
            
            # API Status
            st.subheader("🔌 API Status")
            tmdb_status = "🟢 Active" if os.getenv('TMDB_API_KEY') else "🔴 Inactive"
            llm_status = "🟢 Active" if os.getenv('GROQ_API_KEY') else "🔴 Inactive"
            st.write(f"TMDB: {tmdb_status}")
            st.write(f"AI Insights: {llm_status}")
    
    def _display_emotion_results(self, emotion, confidence_scores):
        """Display emotion detection results with confidence scores"""
        
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
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Detected Emotion")
            emoji = emoji_map.get(emotion, '🎭')
            st.markdown(f"# {emoji}")
            st.markdown(f"## **{emotion.upper()}**")
            st.metric("Confidence", f"{confidence_scores[emotion]*100:.1f}%")
            
            # AI Insight about the emotion
            llm = self._load_llm_recommender()
            if llm:
                with st.spinner("Generating AI insight..."):
                    insight = llm.generate_emotion_insight(emotion, confidence_scores[emotion])
                    st.markdown(f'<div class="ai-insight">💭 <strong>AI Insight:</strong><br>{insight}</div>', 
                              unsafe_allow_html=True)
        
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
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(conf, text=f"**{emo.capitalize()}**")
                with col_b:
                    st.write(f"{conf*100:.1f}%")
    
    def _display_movie_recommendations(self, emotion):
        """Display movie recommendations with TMDB data and AI explanations"""
        st.divider()
        st.header("🎬 Personalized Movie Recommendations")
        
        # Load APIs
        tmdb = self._load_tmdb_client()
        llm = self._load_llm_recommender()
        
        if not tmdb:
            st.error("TMDB API not configured. Please add TMDB_API_KEY to .env file")
            return
        
        # Get movies from TMDB
        with st.spinner("Finding perfect movies for you..."):
            movies = tmdb.get_movies_by_emotion(emotion, count=6)
        
        if not movies:
            st.warning("No movies found. Please try again.")
            return
        
        # Display movies in grid
        for i in range(0, len(movies), 3):
            cols = st.columns(3)
            
            for j, col in enumerate(cols):
                if i + j < len(movies):
                    movie = movies[i + j]
                    
                    with col:
                        # Movie poster
                        if movie['poster_path']:
                            st.image(movie['poster_path'], use_container_width=True)
                        else:
                            st.info("🎬 No poster available")
                        
                        # Movie title and rating
                        st.subheader(movie['title'])
                        
                        # Rating and year
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Rating", f"⭐ {movie['rating']}/10")
                        with col_b:
                            year = movie['release_date'][:4] if movie['release_date'] else 'N/A'
                            st.write(f"📅 **{year}**")
                        
                        # Genres
                        if movie['genres']:
                            genres_text = " • ".join(movie['genres'][:3])
                            st.caption(genres_text)
                        
                        # Overview
                        with st.expander("📖 Overview"):
                            st.write(movie['overview'])
                        
                        # AI Explanation
                        if llm:
                            with st.expander("🤖 Why this movie?"):
                                with st.spinner("Generating..."):
                                    explanation = llm.generate_movie_explanation(
                                        movie['title'], 
                                        emotion, 
                                        movie['overview']
                                    )
                                    st.write(explanation)
                        
                        # Trailer button
                        if movie['id']:
                            videos = tmdb.get_movie_videos(movie['id'])
                            if videos:
                                trailer = next((v for v in videos if v['type'] == 'Trailer'), videos[0])
                                st.link_button("🎥 Watch Trailer", trailer['url'])
                        
                        st.divider()
    
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
            
            # Create columns for image and processing
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
                - 🎭 Real-time emotion detection with confidence scores
                - 🎬 Movie recommendations from TMDB database
                - 🖼️ High-quality movie posters and images
                - ⭐ Ratings, reviews, and release information
                - 🤖 AI-powered personalized explanations
                - 🎥 Direct links to movie trailers
                """
            )


def main():
    """Main entry point for Streamlit app"""
    app = EmotionApp()
    app.run()


if __name__ == "__main__":
    main()
