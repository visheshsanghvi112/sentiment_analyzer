#!/usr/bin/env python3
"""
Simple test script to check module imports
"""

import sys
import traceback

def test_basic_imports():
    """Test basic Python imports"""
    print("🔍 Testing basic imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except Exception as e:
        print(f"❌ Error importing pandas: {e}")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except Exception as e:
        print(f"❌ Error importing numpy: {e}")
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except Exception as e:
        print(f"❌ Error importing streamlit: {e}")

def test_ml_imports():
    """Test machine learning imports"""
    print("\n🧠 Testing ML imports...")
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ scikit-learn imported successfully")
    except Exception as e:
        print(f"❌ Error importing scikit-learn: {e}")
    
    try:
        from rapidfuzz import fuzz, process
        print("✅ rapidfuzz imported successfully")
    except Exception as e:
        print(f"❌ Error importing rapidfuzz: {e}")

def test_nlp_imports():
    """Test NLP imports"""
    print("\n📝 Testing NLP imports...")
    
    try:
        from langdetect import detect
        print("✅ langdetect imported successfully")
    except Exception as e:
        print(f"❌ Error importing langdetect: {e}")
    
    try:
        from googletrans import Translator
        print("✅ googletrans imported successfully")
    except Exception as e:
        print(f"❌ Error importing googletrans: {e}")

def test_transformers():
    """Test transformers imports"""
    print("\n🤖 Testing transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers imported successfully")
    except Exception as e:
        print(f"❌ Error importing sentence-transformers: {e}")
    
    try:
        from transformers import pipeline
        print("✅ transformers imported successfully")
    except Exception as e:
        print(f"❌ Error importing transformers: {e}")

def test_existing_modules():
    """Test existing modules"""
    print("\n📦 Testing existing modules...")
    
    try:
        from sentiment_utils import analyze_sentiment
        print("✅ sentiment_utils imported successfully")
    except Exception as e:
        print(f"❌ Error importing sentiment_utils: {e}")
    
    try:
        from gemini_api import get_gemini_insight
        print("✅ gemini_api imported successfully")
    except Exception as e:
        print(f"❌ Error importing gemini_api: {e}")

def test_new_modules():
    """Test new modules"""
    print("\n🆕 Testing new modules...")
    
    try:
        from search_utils import MovieSearchEngine
        print("✅ search_utils.MovieSearchEngine imported successfully")
    except Exception as e:
        print(f"❌ Error importing search_utils.MovieSearchEngine: {e}")
    
    try:
        from search_utils import EmbeddingEngine
        print("✅ search_utils.EmbeddingEngine imported successfully")
    except Exception as e:
        print(f"❌ Error importing search_utils.EmbeddingEngine: {e}")
    
    try:
        from search_utils import TextProcessor
        print("✅ search_utils.TextProcessor imported successfully")
    except Exception as e:
        print(f"❌ Error importing search_utils.TextProcessor: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting simple import tests...\n")
    
    test_basic_imports()
    test_ml_imports()
    test_nlp_imports()
    test_transformers()
    test_existing_modules()
    test_new_modules()
    
    print("\n✅ Import tests completed!")

if __name__ == "__main__":
    main() 