import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Try to import transformers, fall back gracefully if not available
TRANSFORMERS_AVAILABLE = False
pipeline = None
AutoTokenizer = None
AutoModelForSequenceClassification = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"⚠️ Transformers not available: {e}")
    print("📝 Using VADER sentiment analysis only")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    from vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re
from collections import Counter
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# Download required NLTK data with error handling
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass  # Continue without punkt if download fails

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass  # Continue without stopwords if download fails

# Call this function to ensure data is available
ensure_nltk_data()

# Initialize models with better error handling
@st.cache_resource
def load_sentiment_model():
    """Load the pretrained sentiment analysis model"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        return pipeline("sentiment-analysis", 
                       model="distilbert-base-uncased-finetuned-sst-2-english",
                       return_all_scores=True)
    except Exception as e:
        st.error(f"❌ Error loading sentiment model: {e}")
        return None

@st.cache_resource
def load_vader_analyzer():
    """Load VADER sentiment analyzer"""
    try:
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"❌ Error loading VADER analyzer: {e}")
        return None

def analyze_sentiment(text, method="transformers"):
    """
    Analyze sentiment of text using either transformers or VADER
    
    Args:
        text (str): Text to analyze
        method (str): Either 'transformers' or 'vader'
    
    Returns:
        tuple: (sentiment, confidence/score)
    """
    if method == "transformers":
        if not TRANSFORMERS_AVAILABLE:
            # Fall back to VADER if transformers not available
            return analyze_sentiment(text, method="vader")
            
        model = load_sentiment_model()
        if not model:
            return "ERROR", 0.0
            
        try:
            result = model(text)[0]
            
            # Get the prediction with highest score
            sentiment_label = max(result, key=lambda x: x['score'])
            sentiment = sentiment_label['label']
            confidence = sentiment_label['score']
            
            return sentiment, confidence
        except Exception as e:
            st.error(f"Error in transformer analysis: {e}")
            return "ERROR", 0.0
    
    elif method == "vader":
        analyzer = load_vader_analyzer()
        if not analyzer:
            return "ERROR", 0.0
            
        try:
            scores = analyzer.polarity_scores(text)
            compound_score = scores['compound']
            
            # Determine sentiment based on compound score
            if compound_score >= 0.05:
                sentiment = "POSITIVE"
            elif compound_score <= -0.05:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
                
            return sentiment, abs(compound_score)
        except Exception as e:
            st.error(f"Error in VADER analysis: {e}")
            return "ERROR", 0.0

def detect_emotion(text):
    """
    Simple emotion detection using keyword matching
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: Detected emotion
    """
    text_lower = text.lower()
    
    # Define emotion keywords
    emotion_keywords = {
        'joy': ['happy', 'joy', 'excited', 'amazing', 'wonderful', 'fantastic', 'great', 'excellent', 'love', 'brilliant'],
        'anger': ['angry', 'hate', 'terrible', 'awful', 'disgusting', 'horrible', 'worst', 'rage', 'furious', 'annoying'],
        'sadness': ['sad', 'depressed', 'disappointed', 'boring', 'dull', 'tragic', 'miserable', 'gloomy'],
        'fear': ['scary', 'frightening', 'terrifying', 'creepy', 'horror', 'afraid', 'nervous', 'anxious'],
        'surprise': ['surprising', 'unexpected', 'shocking', 'amazing', 'incredible', 'unbelievable'],
        'disgust': ['disgusting', 'gross', 'revolting', 'sick', 'nasty', 'repulsive']
    }
    
    emotion_scores = {}
    
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        emotion_scores[emotion] = score
    
    # Return emotion with highest score, or 'neutral' if no emotions detected
    if max(emotion_scores.values()) == 0:
        return 'neutral'
    
    return max(emotion_scores, key=emotion_scores.get)

def get_sentiment_distribution(df):
    """
    Create a bar chart showing sentiment distribution
    
    Args:
        df (pd.DataFrame): DataFrame with 'label' column (0=negative, 1=positive)
    
    Returns:
        matplotlib.figure.Figure: Sentiment distribution plot
    """
    # Convert numeric labels to text
    sentiment_counts = df['label'].map({0: 'Negative', 1: 'Positive'}).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    
    ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def clean_text(text):
    """Clean text for word cloud generation"""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def get_wordcloud(df):
    """
    Generate word cloud from review texts
    
    Args:
        df (pd.DataFrame): DataFrame with 'text' column
    
    Returns:
        matplotlib.figure.Figure: Word cloud plot
    """
    # Combine all text
    all_text = ' '.join(df['text'].astype(str))
    cleaned_text = clean_text(all_text)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(cleaned_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Reviews', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def get_top_words(df, sentiment_label, top_n=10):
    """
    Get top words for a specific sentiment
    
    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'label' columns
        sentiment_label (int): 0 for negative, 1 for positive
        top_n (int): Number of top words to return
    
    Returns:
        list: Top words for the sentiment
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    stop_words = set(stopwords.words('english'))
    
    # Filter by sentiment
    sentiment_texts = df[df['label'] == sentiment_label]['text']
    
    # Tokenize and clean
    all_words = []
    for text in sentiment_texts:
        cleaned_text = clean_text(str(text))
        tokens = word_tokenize(cleaned_text)
        words = [word for word in tokens if word not in stop_words and len(word) > 2]
        all_words.extend(words)
    
    # Get most common words
    word_freq = Counter(all_words)
    return word_freq.most_common(top_n)
