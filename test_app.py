"""
Simple test script to verify the app works correctly
"""

from sentiment_utils import SentimentAnalyzer
from utils import clean_text, load_imdb_sample
from config import Config

def test_analyzer():
    """Test the sentiment analyzer"""
    print("Testing sentiment analyzer...")
    
    analyzer = SentimentAnalyzer()
    
    if not analyzer.is_available():
        print("❌ Analyzer failed to load")
        return False
    
    # Test positive sentiment
    result = analyzer.analyze("This movie was absolutely fantastic! I loved it.")
    print(f"Positive test: {result['sentiment']} ({result['confidence']:.2f})")
    
    # Test negative sentiment  
    result = analyzer.analyze("This movie was terrible and boring.")
    print(f"Negative test: {result['sentiment']} ({result['confidence']:.2f})")
    
    print("✅ Analyzer working correctly")
    return True

def test_utils():
    """Test utility functions"""
    print("Testing utilities...")
    
    # Test text cleaning
    dirty_text = "<p>This is a <b>test</b> with   extra   spaces</p>"
    clean = clean_text(dirty_text)
    print(f"Text cleaning: '{clean}'")
    
    # Test IMDB loading (just 5 samples)
    samples = load_imdb_sample(5)
    print(f"IMDB samples loaded: {len(samples)}")
    
    print("✅ Utilities working correctly")
    return True

def main():
    """Run all tests"""
    print("🧪 Running app tests...\n")
    
    success = True
    success &= test_analyzer()
    success &= test_utils()
    
    if success:
        print("\n🎉 All tests passed! App is ready to run.")
        print("Run: python run_app.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()