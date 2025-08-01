#!/usr/bin/env python3
"""
Quick test script to verify all components work
Run this before deploying to catch any issues
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✅ Datasets imported")
    except ImportError as e:
        print(f"❌ Datasets import failed: {e}")
        return False
    
    try:
        from transformers import pipeline
        print("✅ Transformers imported")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Gemini API imported")
    except ImportError as e:
        print(f"❌ Gemini API import failed: {e}")
        return False
    
    try:
        from vaderSentiment import SentimentIntensityAnalyzer
        print("✅ VADER imported")
    except ImportError as e:
        print(f"❌ VADER import failed: {e}")
        return False
        
    return True

def test_basic_functionality():
    """Test basic functionality without Streamlit"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test VADER
        from vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores("This movie is great!")
        print(f"✅ VADER test: {score}")
    except Exception as e:
        print(f"❌ VADER test failed: {e}")
        return False
    
    try:
        # Test transformers
        from transformers import pipeline
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = classifier("This movie is amazing!")
        print(f"✅ Transformer test: {result}")
    except Exception as e:
        print(f"❌ Transformer test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🎬 IMDB Sentiment Analysis App - Production Test")
    print("=" * 50)
    
    if not test_imports():
        print("\n❌ Import tests failed. Please install requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed.")
        return False
    
    print("\n🎉 All tests passed! App is ready for production.")
    print("\n📋 Deployment checklist:")
    print("   ✅ All dependencies installed")
    print("   ✅ Core functionality working")
    print("   🔑 Remember to add GEMINI_API_KEY in Streamlit secrets")
    print("   🚀 Ready to deploy!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
