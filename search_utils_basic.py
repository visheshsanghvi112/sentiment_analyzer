import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from typing import List, Tuple, Dict, Optional
import streamlit as st
import re
from langdetect import detect, LangDetectException
from googletrans import Translator
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class MovieSearchEngine:
    """Fast movie search engine with autocomplete and fuzzy matching"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.movie_names = self._extract_movie_names()
        self.movie_name_to_reviews = self._create_movie_mapping()
        
    def _extract_movie_names(self) -> List[str]:
        """Extract unique movie names from the dataset"""
        movie_names = set()
        
        # Extract movie names from review text using common patterns
        for text in self.df['text'].astype(str):
            # Look for patterns like "movie name (year)" or "movie name"
            movie_patterns = [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(\d{4}\)',  # Movie (Year)
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+a',     # Movie is a
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+',     # Movie was
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+',     # Movie has
            ]
            
            for pattern in movie_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match.split()) >= 1:  # At least one word
                        movie_names.add(match.strip())
        
        # Add some common movie names that might be missed
        common_movies = [
            "Titanic", "Avatar", "The Godfather", "Star Wars", "Jaws", 
            "E.T.", "Jurassic Park", "Forrest Gump", "The Matrix", "Pulp Fiction",
            "Fight Club", "The Shawshank Redemption", "Goodfellas", "The Silence of the Lambs"
        ]
        movie_names.update(common_movies)
        
        return sorted(list(movie_names))
    
    def _create_movie_mapping(self) -> Dict[str, pd.DataFrame]:
        """Create mapping from movie names to their reviews"""
        mapping = {}
        
        for movie_name in self.movie_names:
            # Find reviews that mention this movie
            mask = self.df['text'].str.contains(movie_name, case=False, na=False)
            if mask.sum() > 0:
                mapping[movie_name] = self.df[mask].copy()
        
        return mapping
    
    def autocomplete(self, query: str, limit: int = 10) -> List[str]:
        """
        Fast autocomplete with fuzzy matching
        
        Args:
            query (str): User input
            limit (int): Maximum number of suggestions
            
        Returns:
            List[str]: List of matching movie names
        """
        if not query or len(query) < 2:
            return []
        
        start_time = time.time()
        
        # Use RapidFuzz for fast fuzzy matching
        matches = process.extract(
            query, 
            self.movie_names,
            scorer=fuzz.partial_ratio,
            limit=limit * 2  # Get more candidates for filtering
        )
        
        # Filter and sort by relevance
        filtered_matches = []
        for name, score in matches:
            if score >= 60:  # Minimum similarity threshold
                filtered_matches.append((name, score))
        
        # Sort by score and return top matches
        filtered_matches.sort(key=lambda x: x[1], reverse=True)
        results = [name for name, score in filtered_matches[:limit]]
        
        # Performance check
        elapsed = time.time() - start_time
        if elapsed > 0.2:  # Log slow queries
            st.warning(f"⚠️ Autocomplete took {elapsed:.3f}s (target: <0.2s)")
        
        return results
    
    def fuzzy_search(self, query: str, threshold: int = 70) -> List[Tuple[str, int]]:
        """
        Fuzzy search for movie names with similarity scores
        
        Args:
            query (str): Search query
            threshold (int): Minimum similarity score (0-100)
            
        Returns:
            List[Tuple[str, int]]: List of (movie_name, similarity_score)
        """
        if not query:
            return []
        
        matches = process.extract(
            query,
            self.movie_names,
            scorer=fuzz.ratio,
            limit=20
        )
        
        # Filter by threshold
        return [(name, score) for name, score in matches if score >= threshold]
    
    def get_movie_reviews(self, movie_name: str, limit: int = 30) -> pd.DataFrame:
        """
        Get reviews for a specific movie
        
        Args:
            movie_name (str): Name of the movie
            limit (int): Maximum number of reviews to return
            
        Returns:
            pd.DataFrame: Reviews for the movie
        """
        if movie_name in self.movie_name_to_reviews:
            return self.movie_name_to_reviews[movie_name].head(limit)
        
        # Fallback: search in text
        mask = self.df['text'].str.contains(movie_name, case=False, na=False)
        return self.df[mask].head(limit)
    
    def get_movie_stats(self, movie_name: str) -> Dict:
        """
        Get statistics for a movie
        
        Args:
            movie_name (str): Name of the movie
            
        Returns:
            Dict: Movie statistics
        """
        reviews = self.get_movie_reviews(movie_name)
        
        if reviews.empty:
            return {
                'total_reviews': 0,
                'positive_reviews': 0,
                'negative_reviews': 0,
                'positive_ratio': 0.0,
                'avg_sentiment_score': 0.0
            }
        
        positive_count = (reviews['label'] == 1).sum()
        negative_count = (reviews['label'] == 0).sum()
        total_count = len(reviews)
        
        return {
            'total_reviews': total_count,
            'positive_reviews': positive_count,
            'negative_reviews': negative_count,
            'positive_ratio': positive_count / total_count if total_count > 0 else 0,
            'avg_sentiment_score': reviews['label'].mean()
        }

class TFIDFEmbeddingEngine:
    """TF-IDF based embedding engine for recommendation system"""
    
    def __init__(self):
        self.vectorizer = None
        self.embeddings_cache = {}
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get TF-IDF embeddings for a list of texts
        
        Args:
            texts (List[str]): List of text strings
            
        Returns:
            np.ndarray: TF-IDF embeddings matrix
        """
        # Create cache key
        cache_key = hash(tuple(texts))
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        try:
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            
            embeddings = self.vectorizer.fit_transform(texts).toarray()
            self.embeddings_cache[cache_key] = embeddings
            return embeddings
        except Exception as e:
            st.error(f"❌ Error generating TF-IDF embeddings: {e}")
            return np.array([])
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using TF-IDF
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        embeddings = self.get_embeddings([text1, text2])
        if len(embeddings) < 2:
            return 0.0
        
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

class TextProcessor:
    """Handles text preprocessing and language detection"""
    
    def __init__(self):
        self.translator = Translator()
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Language code (e.g., 'en', 'es', 'fr')
        """
        try:
            return detect(text)
        except LangDetectException:
            return 'en'  # Default to English
    
    def translate_text(self, text: str, target_lang: str = 'en') -> str:
        """
        Translate text to target language
        
        Args:
            text (str): Text to translate
            target_lang (str): Target language code
            
        Returns:
            str: Translated text
        """
        try:
            detected_lang = self.detect_language(text)
            if detected_lang == target_lang:
                return text
            
            result = self.translator.translate(text, dest=target_lang)
            return result.text
        except Exception as e:
            st.warning(f"⚠️ Translation failed: {e}")
            return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for processing
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        return text.strip()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text (str): Text to analyze
            max_keywords (int): Maximum number of keywords
            
        Returns:
            List[str]: List of keywords
        """
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(words)
        
        return [word for word, freq in word_freq.most_common(max_keywords)] 