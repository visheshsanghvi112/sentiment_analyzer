import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import time
from search_utils_basic import TFIDFEmbeddingEngine, TextProcessor

class MovieRecommender:
    """Content-based movie recommendation system using TF-IDF"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.embedding_engine = TFIDFEmbeddingEngine()
        self.text_processor = TextProcessor()
        self.movie_embeddings = {}
        self.review_embeddings = {}
        self.similarity_matrix = None
        
    def prepare_movie_data(self, movie_name: str) -> pd.DataFrame:
        """
        Prepare movie data for recommendation
        
        Args:
            movie_name (str): Name of the movie
            
        Returns:
            pd.DataFrame: Movie reviews and metadata
        """
        # Get reviews for the movie
        movie_reviews = self.df[self.df['text'].str.contains(movie_name, case=False, na=False)]
        
        if movie_reviews.empty:
            return pd.DataFrame()
        
        # Add movie name column
        movie_reviews = movie_reviews.copy()
        movie_reviews['movie_name'] = movie_name
        
        return movie_reviews
    
    def get_movie_embedding(self, movie_name: str) -> np.ndarray:
        """
        Get TF-IDF embedding for a movie based on its reviews
        
        Args:
            movie_name (str): Name of the movie
            
        Returns:
            np.ndarray: Movie embedding
        """
        if movie_name in self.movie_embeddings:
            return self.movie_embeddings[movie_name]
        
        movie_reviews = self.prepare_movie_data(movie_name)
        
        if movie_reviews.empty:
            return np.array([])
        
        # Combine all reviews for the movie
        combined_text = ' '.join(movie_reviews['text'].astype(str))
        cleaned_text = self.text_processor.clean_text(combined_text)
        
        # Get embedding
        embedding = self.embedding_engine.get_embeddings([cleaned_text])
        
        if len(embedding) > 0:
            self.movie_embeddings[movie_name] = embedding[0]
            return embedding[0]
        
        return np.array([])
    
    def find_similar_movies(self, movie_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar movies based on review content using TF-IDF
        
        Args:
            movie_name (str): Name of the reference movie
            top_k (int): Number of similar movies to return
            
        Returns:
            List[Tuple[str, float]]: List of (movie_name, similarity_score)
        """
        start_time = time.time()
        
        # Get reference movie embedding
        reference_embedding = self.get_movie_embedding(movie_name)
        
        if len(reference_embedding) == 0:
            return []
        
        # Get all available movies (from search engine)
        from search_utils_basic import MovieSearchEngine
        search_engine = MovieSearchEngine(self.df)
        all_movies = search_engine.movie_names
        
        similarities = []
        
        # Calculate similarities with other movies
        for other_movie in all_movies:
            if other_movie == movie_name:
                continue
                
            other_embedding = self.get_movie_embedding(other_movie)
            
            if len(other_embedding) > 0:
                # Calculate cosine similarity
                similarity = cosine_similarity([reference_embedding], [other_embedding])[0][0]
                similarities.append((other_movie, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        elapsed = time.time() - start_time
        if elapsed > 1.0:  # Log slow recommendations
            st.warning(f"⚠️ Recommendation took {elapsed:.3f}s")
        
        return similarities[:top_k]
    
    def get_mood_based_recommendations(self, mood_description: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get movie recommendations based on mood description using TF-IDF
        
        Args:
            mood_description (str): Description of desired mood (e.g., "feel-good", "dark")
            top_k (int): Number of recommendations to return
            
        Returns:
            List[Tuple[str, float]]: List of (movie_name, similarity_score)
        """
        # Get embedding for mood description
        mood_embedding = self.embedding_engine.get_embeddings([mood_description])
        
        if len(mood_embedding) == 0:
            return []
        
        # Get all available movies
        from search_utils_basic import MovieSearchEngine
        search_engine = MovieSearchEngine(self.df)
        all_movies = search_engine.movie_names
        
        similarities = []
        
        # Calculate similarities with movies
        for movie in all_movies:
            movie_embedding = self.get_movie_embedding(movie)
            
            if len(movie_embedding) > 0:
                similarity = cosine_similarity([mood_embedding[0]], [movie_embedding])[0][0]
                similarities.append((movie, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_review_based_recommendations(self, review_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get movie recommendations based on a review text using TF-IDF
        
        Args:
            review_text (str): Review text to base recommendations on
            top_k (int): Number of recommendations to return
            
        Returns:
            List[Tuple[str, float]]: List of (movie_name, similarity_score)
        """
        # Clean and get embedding for review
        cleaned_review = self.text_processor.clean_text(review_text)
        review_embedding = self.embedding_engine.get_embeddings([cleaned_review])
        
        if len(review_embedding) == 0:
            return []
        
        # Get all available movies
        from search_utils_basic import MovieSearchEngine
        search_engine = MovieSearchEngine(self.df)
        all_movies = search_engine.movie_names
        
        similarities = []
        
        # Calculate similarities with movies
        for movie in all_movies:
            movie_embedding = self.get_movie_embedding(movie)
            
            if len(movie_embedding) > 0:
                similarity = cosine_similarity([review_embedding[0]], [movie_embedding])[0][0]
                similarities.append((movie, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_movie_clusters(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster movies based on their TF-IDF embeddings
        
        Args:
            n_clusters (int): Number of clusters to create
            
        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster_id to movie names
        """
        from sklearn.cluster import KMeans
        
        # Get all movie embeddings
        from search_utils_basic import MovieSearchEngine
        search_engine = MovieSearchEngine(self.df)
        all_movies = search_engine.movie_names
        
        embeddings = []
        valid_movies = []
        
        for movie in all_movies:
            embedding = self.get_movie_embedding(movie)
            if len(embedding) > 0:
                embeddings.append(embedding)
                valid_movies.append(movie)
        
        if len(embeddings) < n_clusters:
            return {}
        
        # Perform clustering
        embeddings_array = np.array(embeddings)
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
        clusters = kmeans.fit_predict(embeddings_array)
        
        # Group movies by cluster
        movie_clusters = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in movie_clusters:
                movie_clusters[cluster_id] = []
            movie_clusters[cluster_id].append(valid_movies[i])
        
        return movie_clusters
    
    def get_movie_features(self, movie_name: str) -> Dict:
        """
        Extract features for a movie based on its reviews
        
        Args:
            movie_name (str): Name of the movie
            
        Returns:
            Dict: Movie features
        """
        movie_reviews = self.prepare_movie_data(movie_name)
        
        if movie_reviews.empty:
            return {}
        
        # Combine all reviews
        combined_text = ' '.join(movie_reviews['text'].astype(str))
        cleaned_text = self.text_processor.clean_text(combined_text)
        
        # Extract features
        features = {
            'keywords': self.text_processor.extract_keywords(cleaned_text, max_keywords=15),
            'avg_sentiment': movie_reviews['label'].mean(),
            'positive_ratio': (movie_reviews['label'] == 1).mean(),
            'review_count': len(movie_reviews),
            'avg_review_length': movie_reviews['text'].str.len().mean(),
            'language': self.text_processor.detect_language(cleaned_text)
        }
        
        return features
    
    def explain_recommendation(self, source_movie: str, target_movie: str) -> str:
        """
        Generate explanation for why a movie is recommended
        
        Args:
            source_movie (str): Original movie
            target_movie (str): Recommended movie
            
        Returns:
            str: Explanation of the recommendation
        """
        source_features = self.get_movie_features(source_movie)
        target_features = self.get_movie_features(target_movie)
        
        if not source_features or not target_features:
            return "Unable to generate explanation due to insufficient data."
        
        # Calculate similarity
        similarity = self.embedding_engine.compute_similarity(
            ' '.join(source_features.get('keywords', [])),
            ' '.join(target_features.get('keywords', []))
        )
        
        # Generate explanation
        explanation_parts = []
        
        # Sentiment similarity
        sentiment_diff = abs(source_features['avg_sentiment'] - target_features['avg_sentiment'])
        if sentiment_diff < 0.2:
            explanation_parts.append("Both movies have similar audience sentiment")
        
        # Keyword overlap
        source_keywords = set(source_features.get('keywords', []))
        target_keywords = set(target_features.get('keywords', []))
        keyword_overlap = len(source_keywords.intersection(target_keywords))
        
        if keyword_overlap > 0:
            common_keywords = source_keywords.intersection(target_keywords)
            explanation_parts.append(f"Both movies share themes like: {', '.join(list(common_keywords)[:3])}")
        
        # Overall similarity
        if similarity > 0.7:
            explanation_parts.append("The movies have very similar emotional and thematic content")
        elif similarity > 0.5:
            explanation_parts.append("The movies share some thematic similarities")
        
        if not explanation_parts:
            explanation_parts.append("The movies have some underlying similarities in their review patterns")
        
        return " ".join(explanation_parts) + f" (similarity: {similarity:.2f})" 