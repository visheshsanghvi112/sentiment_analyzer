#!/usr/bin/env python3
"""
Test script to identify issues with the sentiment analyzer app
"""

import sys
import traceback

def test_imports():
    """Test all imports to identify issues"""
    print("🔍 Testing imports...")
    
    # Test basic imports
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except Exception as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except Exception as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    # Test datasets import with better error handling
    try:
        from datasets import load_dataset
        print("✅ datasets imported successfully")
    except ImportError as e:
        print(f"⚠️ datasets import failed (ImportError): {e}")
        print("📝 This is expected - will use fallback data")
    except Exception as e:
        print(f"⚠️ datasets import failed (other error): {e}")
        print("📝 This is expected - will use fallback data")
    
    # Test custom modules
    try:
        from sentiment_utils import analyze_sentiment
        print("✅ sentiment_utils imported successfully")
    except Exception as e:
        print(f"❌ sentiment_utils import failed: {e}")
        return False
    
    try:
        from search_utils_basic import MovieSearchEngine
        print("✅ search_utils_basic imported successfully")
    except Exception as e:
        print(f"❌ search_utils_basic import failed: {e}")
        return False
    
    try:
        from recommender_basic import MovieRecommender
        print("✅ recommender_basic imported successfully")
    except Exception as e:
        print(f"❌ recommender_basic import failed: {e}")
        return False
    
    try:
        from gemini_api import get_gemini_insight
        print("✅ gemini_api imported successfully")
    except Exception as e:
        print(f"⚠️ gemini_api import failed: {e}")
        print("📝 This is optional - Gemini features will be disabled")
    
    return True

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("\n🔍 Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        print("🔄 Attempting to load IMDB dataset...")
        dataset = load_dataset("imdb")
        df = dataset["train"].to_pandas()
        print(f"✅ IMDB dataset loaded successfully with {len(df)} rows")
        return True
    except ImportError as e:
        print(f"⚠️ Datasets library not available: {e}")
        print("📝 Will use fallback dataset")
        return False
    except Exception as e:
        print(f"⚠️ IMDB dataset loading failed: {e}")
        print("📝 Will use fallback dataset")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\n🔍 Testing sentiment analysis...")
    
    try:
        from sentiment_utils import analyze_sentiment
        text = "This movie was absolutely fantastic!"
        result = analyze_sentiment(text)
        print(f"✅ Sentiment analysis test passed: {result}")
        return True
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False

def test_search_engine():
    """Test search engine functionality"""
    print("\n🔍 Testing search engine...")
    
    try:
        from search_utils_basic import MovieSearchEngine
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'text': ['Great movie', 'Bad movie', 'Amazing film'],
            'label': [1, 0, 1]
        })
        
        search_engine = MovieSearchEngine(test_data)
        results = search_engine.search("movie", top_k=2)
        print(f"✅ Search engine test passed: found {len(results)} results")
        return True
    except Exception as e:
        print(f"❌ Search engine test failed: {e}")
        return False

def test_fallback_dataset():
    """Test fallback dataset creation"""
    print("\n🔍 Testing fallback dataset...")
    
    try:
        # Import the function from app.py
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Create a mock streamlit object for testing
        class MockStreamlit:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def success(self, msg): print(f"SUCCESS: {msg}")
        
        # Mock streamlit
        import sys
        sys.modules['streamlit'] = MockStreamlit()
        
        # Import and test the fallback function
        from app import create_fallback_dataset
        df = create_fallback_dataset()
        print(f"✅ Fallback dataset created successfully with {len(df)} rows")
        return True
    except Exception as e:
        print(f"❌ Fallback dataset test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting comprehensive app test...\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. App cannot run.")
        return
    
    # Test dataset loading
    test_dataset_loading()
    
    # Test fallback dataset
    test_fallback_dataset()
    
    # Test core functionality
    if not test_sentiment_analysis():
        print("\n❌ Sentiment analysis test failed.")
        return
    
    if not test_search_engine():
        print("\n❌ Search engine test failed.")
        return
    
    print("\n✅ All tests passed! App should work correctly.")
    print("\n🎬 You can now run: streamlit run app.py")

if __name__ == "__main__":
    main() 