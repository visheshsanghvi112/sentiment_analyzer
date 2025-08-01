#!/usr/bin/env python3
"""
Test script for the enhanced movie sentiment analyzer
Tests all new modules and functionality
"""

import sys
import traceback
import pandas as pd
from datasets import load_dataset

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from sentiment_utils import analyze_sentiment, get_wordcloud, get_sentiment_distribution, detect_emotion
        print("✅ sentiment_utils imported successfully")
    except Exception as e:
        print(f"❌ Error importing sentiment_utils: {e}")
        return False
    
    try:
        from gemini_api import get_gemini_insight, analyze_review_with_gemini, get_sentiment_explanation
        print("✅ gemini_api imported successfully")
    except Exception as e:
        print(f"❌ Error importing gemini_api: {e}")
        return False
    
    try:
        from search_utils import MovieSearchEngine, EmbeddingEngine, TextProcessor
        print("✅ search_utils imported successfully")
    except Exception as e:
        print(f"❌ Error importing search_utils: {e}")
        return False
    
    try:
        from recommender import MovieRecommender
        print("✅ recommender imported successfully")
    except Exception as e:
        print(f"❌ Error importing recommender: {e}")
        return False
    
    try:
        from gemini_utils import (
            explain_movie_recommendation, analyze_mood_based_search, 
            get_emotional_insights_for_movie, get_movie_comparison_analysis
        )
        print("✅ gemini_utils imported successfully")
    except Exception as e:
        print(f"❌ Error importing gemini_utils: {e}")
        return False
    
    return True

def test_data_loading():
    """Test dataset loading"""
    print("\n📊 Testing data loading...")
    
    try:
        dataset = load_dataset("imdb")
        df = pd.DataFrame(dataset["train"])
        print(f"✅ Dataset loaded successfully: {len(df)} reviews")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def test_search_engine(df):
    """Test search engine functionality"""
    print("\n🔍 Testing search engine...")
    
    try:
        from search_utils import MovieSearchEngine
        search_engine = MovieSearchEngine(df)
        
        # Test autocomplete
        suggestions = search_engine.autocomplete("titanic", limit=5)
        print(f"✅ Autocomplete test: Found {len(suggestions)} suggestions for 'titanic'")
        
        # Test fuzzy search
        fuzzy_matches = search_engine.fuzzy_search("titanic", threshold=60)
        print(f"✅ Fuzzy search test: Found {len(fuzzy_matches)} matches for 'titanic'")
        
        # Test movie stats
        if suggestions:
            movie_stats = search_engine.get_movie_stats(suggestions[0])
            print(f"✅ Movie stats test: Got stats for '{suggestions[0]}'")
        
        return True
    except Exception as e:
        print(f"❌ Error testing search engine: {e}")
        return False

def test_recommender(df):
    """Test recommendation system"""
    print("\n🎯 Testing recommender...")
    
    try:
        from recommender import MovieRecommender
        recommender = MovieRecommender(df)
        
        # Test movie features
        movie_features = recommender.get_movie_features("Titanic")
        if movie_features:
            print(f"✅ Movie features test: Got features for 'Titanic'")
        
        # Test similarity calculation
        from search_utils import MovieSearchEngine
        search_engine = MovieSearchEngine(df)
        movies = search_engine.movie_names[:5]  # Get first 5 movies
        
        if len(movies) >= 2:
            similarity = recommender.embedding_engine.compute_similarity(
                "This is a great movie", "This is an amazing film"
            )
            print(f"✅ Similarity test: Similarity score = {similarity:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing recommender: {e}")
        return False

def test_text_processor():
    """Test text processing functionality"""
    print("\n📝 Testing text processor...")
    
    try:
        from search_utils import TextProcessor
        processor = TextProcessor()
        
        # Test language detection
        lang = processor.detect_language("This is a test")
        print(f"✅ Language detection test: Detected '{lang}'")
        
        # Test text cleaning
        cleaned = processor.clean_text("<html>This is a test!</html>")
        print(f"✅ Text cleaning test: Cleaned text = '{cleaned}'")
        
        # Test keyword extraction
        keywords = processor.extract_keywords("This is a great movie with amazing acting")
        print(f"✅ Keyword extraction test: Found {len(keywords)} keywords")
        
        return True
    except Exception as e:
        print(f"❌ Error testing text processor: {e}")
        return False

def test_embedding_engine():
    """Test embedding engine"""
    print("\n🧠 Testing embedding engine...")
    
    try:
        from search_utils import EmbeddingEngine
        engine = EmbeddingEngine()
        
        # Test embedding generation
        texts = ["This is a great movie", "This is an amazing film"]
        embeddings = engine.get_embeddings(texts)
        
        if len(embeddings) > 0:
            print(f"✅ Embedding test: Generated {len(embeddings)} embeddings")
            
            # Test similarity
            similarity = engine.compute_similarity(texts[0], texts[1])
            print(f"✅ Similarity test: Similarity = {similarity:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing embedding engine: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis"""
    print("\n😊 Testing sentiment analysis...")
    
    try:
        from sentiment_utils import analyze_sentiment, detect_emotion
        
        test_text = "This is a wonderful and amazing movie!"
        
        # Test sentiment analysis
        sentiment, confidence = analyze_sentiment(test_text)
        print(f"✅ Sentiment analysis test: {sentiment} (confidence: {confidence:.3f})")
        
        # Test emotion detection
        emotion = detect_emotion(test_text)
        print(f"✅ Emotion detection test: {emotion}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing sentiment analysis: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting enhanced movie sentiment analyzer tests...\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your dependencies.")
        return False
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        print("\n❌ Data loading failed. Please check your internet connection.")
        return False
    
    # Test search engine
    if not test_search_engine(df):
        print("\n❌ Search engine tests failed.")
        return False
    
    # Test recommender
    if not test_recommender(df):
        print("\n❌ Recommender tests failed.")
        return False
    
    # Test text processor
    if not test_text_processor():
        print("\n❌ Text processor tests failed.")
        return False
    
    # Test embedding engine
    if not test_embedding_engine():
        print("\n❌ Embedding engine tests failed.")
        return False
    
    # Test sentiment analysis
    if not test_sentiment_analysis():
        print("\n❌ Sentiment analysis tests failed.")
        return False
    
    print("\n🎉 All tests passed! The enhanced movie sentiment analyzer is ready to use.")
    print("\n📋 Next steps:")
    print("1. Set up your GEMINI_API_KEY in .env or Streamlit secrets")
    print("2. Run: streamlit run app_enhanced.py")
    print("3. Enjoy the enhanced features!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 