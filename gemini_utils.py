import os
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Load environment variables
load_dotenv()

def configure_gemini():
    """Configure Gemini API with API key"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        # If no API key in .env, try to get from Streamlit secrets
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            st.error("⚠️ Gemini API key not found. Please set GEMINI_API_KEY in your .env file or Streamlit secrets.")
            return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash-exp')

def get_gemini_insight(question, context="movie reviews"):
    """
    Get insights from Gemini AI about movie reviews or sentiment analysis
    
    Args:
        question (str): User's question
        context (str): Context for the question
    
    Returns:
        str: Gemini's response
    """
    try:
        model = configure_gemini()
        if not model:
            return "❌ Cannot connect to Gemini API. Please check your API key."
        
        # Create a comprehensive prompt
        prompt = f"""
        You are an expert in sentiment analysis and movie review analysis. 
        
        Context: {context}
        
        Question: {question}
        
        Please provide a helpful, insightful response about sentiment analysis, movie reviews, or related topics. 
        Keep your response concise but informative (2-3 paragraphs maximum).
        
        If the question is about analyzing a specific review, provide:
        1. Overall sentiment and tone
        2. Key emotional indicators
        3. What makes this review positive/negative
        4. Any interesting linguistic patterns
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Error getting Gemini insight: {str(e)}"

def analyze_review_with_gemini(review_text):
    """
    Get detailed analysis of a specific review using Gemini
    
    Args:
        review_text (str): The review to analyze
    
    Returns:
        str: Detailed analysis from Gemini
    """
    prompt = f"""
    Please analyze this movie review in detail:
    
    "{review_text}"
    
    Provide analysis on:
    1. Overall sentiment (positive/negative/neutral) and confidence
    2. Emotional tone and mood
    3. Specific words or phrases that indicate sentiment
    4. What the reviewer liked or disliked
    5. Writing style and emotional intensity
    
    Keep it concise but thorough.
    """
    
    return get_gemini_insight(prompt, "detailed review analysis")

def get_movie_insights(movie_name, reviews_df):
    """
    Get Gemini insights about a specific movie based on reviews
    
    Args:
        movie_name (str): Name of the movie
        reviews_df (pd.DataFrame): DataFrame containing reviews
    
    Returns:
        str: Insights about the movie
    """
    # Sample a few reviews for context
    sample_reviews = reviews_df.head(3)['text'].tolist()
    reviews_context = "\n\n".join([f"Review {i+1}: {review[:200]}..." for i, review in enumerate(sample_reviews)])
    
    prompt = f"""
    Based on these sample reviews for "{movie_name}":
    
    {reviews_context}
    
    Please provide insights about:
    1. Common themes in the reviews
    2. What audiences generally think about this movie
    3. Most frequently mentioned positive and negative aspects
    4. Overall audience reception patterns
    
    Provide a brief but comprehensive analysis.
    """
    
    return get_gemini_insight(prompt, f"analysis of {movie_name} reviews")

def get_sentiment_explanation(sentiment_result, confidence, review_text):
    """
    Get Gemini explanation for why a review has a certain sentiment
    
    Args:
        sentiment_result (str): The predicted sentiment
        confidence (float): Confidence score
        review_text (str): The original review
    
    Returns:
        str: Explanation of the sentiment analysis
    """
    prompt = f"""
    A sentiment analysis model predicted this review as "{sentiment_result}" with {confidence:.2f} confidence:
    
    "{review_text}"
    
    Please explain:
    1. Why this sentiment classification makes sense
    2. What specific words or phrases support this sentiment
    3. Any nuances or complexities in the review's tone
    4. Whether you agree with the confidence level
    
    Keep it brief but insightful.
    """
    
    return get_gemini_insight(prompt, "sentiment analysis explanation")

def explain_movie_recommendation(source_movie: str, target_movie: str, source_features: Dict, target_features: Dict, similarity_score: float) -> str:
    """
    Generate a detailed explanation for why a movie is recommended using Gemini
    
    Args:
        source_movie (str): Original movie
        target_movie (str): Recommended movie
        source_features (Dict): Features of source movie
        target_features (Dict): Features of target movie
        similarity_score (float): Similarity score between movies
    
    Returns:
        str: Detailed explanation from Gemini
    """
    prompt = f"""
    Explain why "{target_movie}" is recommended for someone who likes "{source_movie}".
    
    Source movie features:
    - Keywords: {', '.join(source_features.get('keywords', []))}
    - Average sentiment: {source_features.get('avg_sentiment', 0):.2f}
    - Positive review ratio: {source_features.get('positive_ratio', 0):.2f}
    
    Target movie features:
    - Keywords: {', '.join(target_features.get('keywords', []))}
    - Average sentiment: {target_features.get('avg_sentiment', 0):.2f}
    - Positive review ratio: {target_features.get('positive_ratio', 0):.2f}
    
    Similarity score: {similarity_score:.2f}
    
    Provide a compelling explanation that covers:
    1. Thematic similarities
    2. Emotional resonance
    3. Audience appeal overlap
    4. Why this recommendation makes sense
    
    Keep it engaging and informative (2-3 paragraphs).
    """
    
    return get_gemini_insight(prompt, "movie recommendation explanation")

def analyze_mood_based_search(mood_description: str, recommended_movies: List[Tuple[str, float]]) -> str:
    """
    Analyze mood-based movie recommendations using Gemini
    
    Args:
        mood_description (str): User's mood description
        recommended_movies (List[Tuple[str, float]]): List of (movie, similarity_score)
    
    Returns:
        str: Analysis of mood-based recommendations
    """
    movies_text = "\n".join([f"- {movie} (similarity: {score:.2f})" for movie, score in recommended_movies])
    
    prompt = f"""
    Analyze these movie recommendations for someone looking for a "{mood_description}" mood:
    
    {movies_text}
    
    Please provide:
    1. How well these movies match the requested mood
    2. What makes each movie suitable for this mood
    3. Any patterns in the recommendations
    4. Suggestions for the user
    
    Keep it helpful and insightful.
    """
    
    return get_gemini_insight(prompt, "mood-based movie analysis")

def get_movie_comparison_analysis(movie1: str, movie2: str, reviews1: pd.DataFrame, reviews2: pd.DataFrame) -> str:
    """
    Compare two movies using Gemini analysis
    
    Args:
        movie1 (str): First movie name
        movie2 (str): Second movie name
        reviews1 (pd.DataFrame): Reviews for first movie
        reviews2 (pd.DataFrame): Reviews for second movie
    
    Returns:
        str: Comparison analysis
    """
    # Sample reviews for comparison
    sample_reviews1 = reviews1.head(2)['text'].tolist()
    sample_reviews2 = reviews2.head(2)['text'].tolist()
    
    reviews1_text = "\n".join([f"Review {i+1}: {review[:150]}..." for i, review in enumerate(sample_reviews1)])
    reviews2_text = "\n".join([f"Review {i+1}: {review[:150]}..." for i, review in enumerate(sample_reviews2)])
    
    prompt = f"""
    Compare these two movies based on their reviews:
    
    {movie1}:
    {reviews1_text}
    
    {movie2}:
    {reviews2_text}
    
    Please analyze:
    1. Similarities and differences in audience reception
    2. Common themes or elements
    3. Emotional tone comparison
    4. Which movie might appeal to different audiences
    5. Overall quality assessment based on reviews
    
    Provide a balanced comparison.
    """
    
    return get_gemini_insight(prompt, f"comparison of {movie1} and {movie2}")

def generate_movie_cluster_insights(movie_clusters: Dict[int, List[str]]) -> str:
    """
    Generate insights about movie clusters using Gemini
    
    Args:
        movie_clusters (Dict[int, List[str]]): Dictionary of movie clusters
    
    Returns:
        str: Cluster analysis insights
    """
    clusters_text = ""
    for cluster_id, movies in movie_clusters.items():
        clusters_text += f"\nCluster {cluster_id + 1}: {', '.join(movies[:5])}"
        if len(movies) > 5:
            clusters_text += f" (and {len(movies) - 5} more)"
    
    prompt = f"""
    Analyze these movie clusters based on their review patterns:
    
    {clusters_text}
    
    Please provide:
    1. What themes or characteristics define each cluster
    2. What makes movies in the same cluster similar
    3. How these clusters might help with movie recommendations
    4. Any interesting patterns you notice
    
    Keep it insightful and useful for understanding movie preferences.
    """
    
    return get_gemini_insight(prompt, "movie cluster analysis")

def get_emotional_insights_for_movie(movie_name: str, reviews_df: pd.DataFrame) -> str:
    """
    Get emotional insights about a movie using Gemini
    
    Args:
        movie_name (str): Name of the movie
        reviews_df (pd.DataFrame): Reviews for the movie
    
    Returns:
        str: Emotional analysis
    """
    # Get sentiment distribution
    positive_count = (reviews_df['label'] == 1).sum()
    negative_count = (reviews_df['label'] == 0).sum()
    total_count = len(reviews_df)
    
    # Sample reviews for emotional analysis
    sample_reviews = reviews_df.head(3)['text'].tolist()
    reviews_text = "\n\n".join([f"Review {i+1}: {review[:200]}..." for i, review in enumerate(sample_reviews)])
    
    prompt = f"""
    Analyze the emotional impact of "{movie_name}" based on these reviews:
    
    Sentiment breakdown:
    - Positive reviews: {positive_count} ({positive_count/total_count*100:.1f}%)
    - Negative reviews: {negative_count} ({negative_count/total_count*100:.1f}%)
    - Total reviews: {total_count}
    
    Sample reviews:
    {reviews_text}
    
    Please provide:
    1. The emotional journey this movie takes viewers on
    2. What emotions it typically evokes
    3. Why it resonates (or doesn't) with audiences
    4. The emotional themes and messages
    5. Who would most enjoy this movie emotionally
    
    Focus on the emotional and psychological aspects.
    """
    
    return get_gemini_insight(prompt, f"emotional analysis of {movie_name}")

def get_movie_recommendation_prompt(mood: str, genre_preference: str = "", previous_movies: List[str] = None) -> str:
    """
    Generate a personalized movie recommendation prompt
    
    Args:
        mood (str): Desired mood
        genre_preference (str): Preferred genre
        previous_movies (List[str]): Previously enjoyed movies
    
    Returns:
        str: Personalized recommendation prompt
    """
    prompt_parts = [f"I'm looking for a movie that gives me a {mood} feeling"]
    
    if genre_preference:
        prompt_parts.append(f"I prefer {genre_preference} movies")
    
    if previous_movies:
        prompt_parts.append(f"I recently enjoyed: {', '.join(previous_movies)}")
    
    prompt_parts.append("What would you recommend and why?")
    
    return " ".join(prompt_parts)

def analyze_review_patterns(reviews_df: pd.DataFrame) -> str:
    """
    Analyze patterns in movie reviews using Gemini
    
    Args:
        reviews_df (pd.DataFrame): DataFrame of reviews
    
    Returns:
        str: Pattern analysis
    """
    # Calculate basic statistics
    total_reviews = len(reviews_df)
    positive_ratio = (reviews_df['label'] == 1).mean()
    avg_length = reviews_df['text'].str.len().mean()
    
    # Sample some reviews
    sample_reviews = reviews_df.head(3)['text'].tolist()
    reviews_text = "\n\n".join([f"Review {i+1}: {review[:150]}..." for i, review in enumerate(sample_reviews)])
    
    prompt = f"""
    Analyze the patterns in these movie reviews:
    
    Statistics:
    - Total reviews: {total_reviews}
    - Positive ratio: {positive_ratio:.2f}
    - Average review length: {avg_length:.0f} characters
    
    Sample reviews:
    {reviews_text}
    
    Please identify:
    1. Common themes and patterns
    2. Writing style characteristics
    3. What makes reviews compelling or not
    4. Sentiment patterns and trends
    5. Any interesting linguistic features
    
    Provide insights that could help understand audience preferences.
    """
    
    return get_gemini_insight(prompt, "review pattern analysis") 