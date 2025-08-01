import os
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

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
    return genai.GenerativeModel('gemini-2.5-pro')

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
