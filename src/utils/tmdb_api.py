"""
TMDB API Integration
Handles movie data retrieval from The Movie Database
"""

import os
import requests
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TMDBClient:
    """Client for interacting with TMDB API"""
    
    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
    
    # Map emotions to TMDB genre IDs and keywords
    EMOTION_TO_GENRES = {
        'anger': {
            'genres': [28, 53, 80],  # Action, Thriller, Crime
            'keywords': 'revenge|justice|fight'
        },
        'disgust': {
            'genres': [27, 53],  # Horror, Thriller
            'keywords': 'dark|disturbing|psychological'
        },
        'fear': {
            'genres': [27, 53, 9648],  # Horror, Thriller, Mystery
            'keywords': 'scary|suspense|thriller'
        },
        'happy': {
            'genres': [35, 10751, 10749],  # Comedy, Family, Romance
            'keywords': 'feel-good|uplifting|heartwarming'
        },
        'sad': {
            'genres': [18, 10749],  # Drama, Romance
            'keywords': 'emotional|touching|heartbreaking'
        },
        'surprise': {
            'genres': [878, 9648, 53],  # Sci-Fi, Mystery, Thriller
            'keywords': 'twist|unexpected|mind-bending'
        },
        'neutral': {
            'genres': [18, 28, 12],  # Drama, Action, Adventure
            'keywords': 'popular|acclaimed|award-winning'
        }
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize TMDB client
        
        Args:
            api_key (str, optional): TMDB API key. If None, reads from env.
        """
        self.api_key = api_key or os.getenv('TMDB_API_KEY')
        
        if not self.api_key:
            raise ValueError("TMDB API key not found. Set TMDB_API_KEY environment variable.")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make API request to TMDB
        
        Args:
            endpoint (str): API endpoint
            params (dict): Query parameters
        
        Returns:
            dict: API response or None if error
        """
        if params is None:
            params = {}
        
        params['api_key'] = self.api_key
        
        try:
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"TMDB API request failed: {e}")
            return None
    
    def get_movies_by_emotion(self, emotion: str, count: int = 10) -> List[Dict]:
        """
        Get movie recommendations based on emotion
        
        Args:
            emotion (str): Detected emotion
            count (int): Number of movies to return
        
        Returns:
            List[Dict]: List of movie data
        """
        emotion = emotion.lower()
        
        if emotion not in self.EMOTION_TO_GENRES:
            emotion = 'neutral'
        
        config = self.EMOTION_TO_GENRES[emotion]
        genre_ids = '|'.join(map(str, config['genres']))
        
        # Discover movies
        params = {
            'with_genres': genre_ids,
            'sort_by': 'vote_average.desc',
            'vote_count.gte': 1000,  # Minimum votes for quality
            'page': 1,
            'language': 'en-US'
        }
        
        data = self._make_request('/discover/movie', params)
        
        if not data or 'results' not in data:
            logger.warning(f"No movies found for emotion: {emotion}")
            return []
        
        movies = []
        for movie in data['results'][:count]:
            movie_data = {
                'id': movie.get('id'),
                'title': movie.get('title', 'Unknown'),
                'overview': movie.get('overview', 'No description available'),
                'poster_path': self._get_poster_url(movie.get('poster_path')),
                'backdrop_path': self._get_poster_url(movie.get('backdrop_path')),
                'release_date': movie.get('release_date', 'N/A'),
                'rating': round(movie.get('vote_average', 0), 1),
                'genres': self._get_genre_names(movie.get('genre_ids', []))
            }
            movies.append(movie_data)
        
        return movies
    
    def _get_poster_url(self, path: Optional[str]) -> Optional[str]:
        """Get full poster URL"""
        if path:
            return f"{self.IMAGE_BASE_URL}{path}"
        return None
    
    def _get_genre_names(self, genre_ids: List[int]) -> List[str]:
        """Convert genre IDs to names"""
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, '') for gid in genre_ids if gid in genre_map]
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific movie
        
        Args:
            movie_id (int): TMDB movie ID
        
        Returns:
            dict: Movie details or None
        """
        data = self._make_request(f'/movie/{movie_id}')
        
        if not data:
            return None
        
        return {
            'id': data.get('id'),
            'title': data.get('title'),
            'overview': data.get('overview'),
            'poster_path': self._get_poster_url(data.get('poster_path')),
            'backdrop_path': self._get_poster_url(data.get('backdrop_path')),
            'release_date': data.get('release_date'),
            'runtime': data.get('runtime'),
            'rating': round(data.get('vote_average', 0), 1),
            'genres': [g['name'] for g in data.get('genres', [])],
            'homepage': data.get('homepage')
        }
    
    def get_movie_videos(self, movie_id: int) -> List[Dict]:
        """Get trailers and videos for a movie"""
        data = self._make_request(f'/movie/{movie_id}/videos')
        
        if not data or 'results' not in data:
            return []
        
        videos = []
        for video in data['results']:
            if video.get('site') == 'YouTube':
                videos.append({
                    'key': video.get('key'),
                    'name': video.get('name'),
                    'type': video.get('type'),
                    'url': f"https://www.youtube.com/watch?v={video.get('key')}"
                })
        
        return videos

