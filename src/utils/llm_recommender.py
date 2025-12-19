"""
LLM-based Movie Recommendation Enhancement
Uses Groq API to generate personalized explanations
"""

import os
from groq import Groq
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMRecommender:
    """Generate personalized movie recommendations using LLM"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize LLM recommender
        
        Args:
            api_key (str, optional): Groq API key. If None, reads from env.
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        if not self.api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "mixtral-8x7b-32768"  # Fast and good quality
    
    def generate_movie_explanation(self, movie_title: str, emotion: str, 
                                   overview: str = None) -> str:
        """
        Generate personalized explanation for why a movie matches the emotion
        
        Args:
            movie_title (str): Movie title
            emotion (str): Detected emotion
            overview (str, optional): Movie overview/description
        
        Returns:
            str: Personalized explanation
        """
        try:
            prompt = self._create_explanation_prompt(movie_title, emotion, overview)
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a movie recommendation expert who provides "
                                 "thoughtful, personalized explanations for movie suggestions. "
                                 "Keep responses concise (2-3 sentences) and empathetic."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=150
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info(f"Generated explanation for {movie_title}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._get_fallback_explanation(emotion)
    
    def _create_explanation_prompt(self, movie_title: str, emotion: str, 
                                   overview: str = None) -> str:
        """Create prompt for explanation generation"""
        base_prompt = (
            f"A user is feeling {emotion}. Explain in 2-3 sentences why "
            f'"{movie_title}" is a perfect match for their current mood.'
        )
        
        if overview:
            base_prompt += f" The movie is about: {overview[:200]}"
        
        base_prompt += " Be warm, understanding, and encouraging."
        
        return base_prompt
    
    def generate_emotion_insight(self, emotion: str, confidence: float) -> str:
        """
        Generate insight about the detected emotion
        
        Args:
            emotion (str): Detected emotion
            confidence (float): Confidence score (0-1)
        
        Returns:
            str: Personalized insight
        """
        try:
            prompt = (
                f"A person is showing {emotion} emotion with {confidence*100:.0f}% confidence. "
                f"Provide a brief, empathetic insight about their mood and what kind of "
                f"entertainment might help them. Keep it to 2 sentences."
            )
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an empathetic mood expert who provides "
                                 "supportive insights about emotions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating insight: {e}")
            return self._get_fallback_insight(emotion)
    
    def _get_fallback_explanation(self, emotion: str) -> str:
        """Fallback explanations if API fails"""
        fallbacks = {
            'anger': "This powerful film channels intense emotions into a compelling narrative that resonates with your current feelings.",
            'disgust': "This thought-provoking film matches the intensity of your emotions with its dark, gripping storyline.",
            'fear': "This suspenseful thriller will engage your heightened senses while providing a safe outlet for your emotions.",
            'happy': "This uplifting film will amplify your positive mood with its heartwarming story and joyful moments.",
            'sad': "This emotionally rich film provides catharsis and understanding for what you're feeling right now.",
            'surprise': "This mind-bending film will channel your heightened awareness into an engaging, unexpected journey.",
            'neutral': "This highly-rated film offers an engaging story that will draw you in and captivate your attention."
        }
        return fallbacks.get(emotion.lower(), "This film is recommended based on your current mood.")
    
    def _get_fallback_insight(self, emotion: str) -> str:
        """Fallback insights if API fails"""
        insights = {
            'anger': "Strong emotions can be powerful motivators. Movies that explore justice and resolution might resonate with you.",
            'disgust': "You're experiencing a strong reaction. Dark, thought-provoking films can help process these feelings.",
            'fear': "Your senses are heightened. Controlled suspense in films can provide a safe outlet for these emotions.",
            'happy': "You're in a great mood! Feel-good movies will amplify your positive energy and joy.",
            'sad': "You're processing deep emotions. Films that validate and explore feelings can provide comfort.",
            'surprise': "You're experiencing heightened awareness. Films with unexpected twists will engage your curious mind.",
            'neutral': "You're in a balanced state. This is perfect for exploring a wide range of compelling stories."
        }
        return insights.get(emotion.lower(), "Movies can be a great way to explore and process emotions.")

