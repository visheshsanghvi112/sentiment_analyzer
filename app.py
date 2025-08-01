from datasets import load_dataset
import pandas as pd
import streamlit as st
import sys
import traceback

# Import custom modules with error handling
try:
    from sentiment_utils import analyze_sentiment, get_wordcloud, get_sentiment_distribution, detect_emotion
except ImportError as e:
    st.error(f"❌ Error importing sentiment_utils: {e}")
    st.stop()

try:
    from gemini_api import get_gemini_insight, analyze_review_with_gemini, get_sentiment_explanation
except ImportError as e:
    st.error(f"❌ Error importing gemini_api: {e}")
    st.stop()

# Load dataset with caching for better performance
@st.cache_data
def load_imdb_data():
    """Load and cache the IMDB dataset"""
    try:
        with st.spinner("🔄 Loading IMDB dataset (this may take a moment on first run)..."):
            dataset = load_dataset("imdb")
            return pd.DataFrame(dataset["train"])
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.info("💡 This might be due to network issues. Please refresh the page or try again later.")
        return pd.DataFrame()

# Initialize dataset
try:
    df = load_imdb_data()
except Exception as e:
    st.error(f"❌ Critical error: {e}")
    st.stop()

def get_reviews_for_movie(movie_name, limit=30):
    filt = df[df["text"].str.contains(movie_name, case=False, na=False)]
    return filt.head(limit).copy()

def main():
    st.set_page_config(
        page_title="IMDB Sentiment Analyzer", 
        layout="wide",
        page_icon="🎬"
    )

    # Sidebar
    st.sidebar.title("🎬 IMDB Sentiment Analyzer")
    st.sidebar.markdown("---")
    
    # Dataset info
    if not df.empty:
        st.sidebar.metric("Total Reviews", f"{len(df):,}")
        positive_count = (df['label'] == 1).sum()
        st.sidebar.metric("Positive Reviews", f"{positive_count:,}")
        st.sidebar.metric("Negative Reviews", f"{len(df) - positive_count:,}")
    
    st.sidebar.markdown("---")

    PAGES = {
        "📊 Dataset Explorer": "explorer",
        "🧠 Analyze Review": "analyze",
        "🤖 Gemini Insights": "gemini"
    }

    page = st.sidebar.selectbox("Select a page", list(PAGES.keys()))
    
    # Add helpful information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This app uses:
    - **IMDB Dataset** from Hugging Face
    - **BERT** for sentiment analysis
    - **VADER** for traditional scoring
    - **Gemini AI** for insights
    """)
    
    st.sidebar.markdown("### 🚀 Features")
    st.sidebar.markdown("""
    - Explore 50K movie reviews
    - Analyze custom reviews
    - AI-powered insights
    - Emotion detection
    - Interactive visualizations
    """)
    
    # Handle empty dataset
    if df.empty:
        st.error("Failed to load IMDB dataset. Please check your internet connection.")
        return

    if PAGES[page] == "explorer":
        st.title("IMDB Dataset Explorer")
        st.write("Browse and filter movie reviews from the IMDB dataset.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            movie_name = st.text_input("Filter by movie name (optional):")
        with col2:
            limit = st.slider("Number of reviews to show", 10, 100, 30)
        
        if movie_name:
            reviews = get_reviews_for_movie(movie_name, limit)
            if len(reviews) == 0:
                st.warning(f"No reviews found containing '{movie_name}'. Showing random reviews instead.")
                reviews = df.sample(limit)
            else:
                st.success(f"Found {len(reviews)} reviews mentioning '{movie_name}'")
        else:
            reviews = df.sample(limit)
        
        # Display reviews in an expandable format
        st.subheader("Reviews")
        for idx, (_, row) in enumerate(reviews.iterrows()):
            sentiment_label = "Positive" if row['label'] == 1 else "Negative"
            with st.expander(f"Review {idx+1} - {sentiment_label}"):
                st.write(row['text'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            try:
                fig = get_sentiment_distribution(reviews)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error creating sentiment chart: {e}")
        
        with col2:
            st.subheader("Word Cloud")
            try:
                wc_fig = get_wordcloud(reviews)
                st.pyplot(wc_fig)
            except Exception as e:
                st.error(f"❌ Error creating word cloud: {e}")

    elif PAGES[page] == "analyze":
        st.title("Analyze Your Own Review")
        user_review = st.text_area("Enter your movie review:", height=150)
        
        if user_review:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Analysis")
                try:
                    sentiment, confidence = analyze_sentiment(user_review)
                    st.write(f"**Sentiment:** {sentiment}")
                    st.write(f"**Confidence:** {confidence:.2f}")
                except Exception as e:
                    st.error(f"❌ Error in sentiment analysis: {e}")
                
                st.subheader("VADER Score")
                try:
                    vader_sentiment, vader_score = analyze_sentiment(user_review, method="vader")
                    st.write(f"**VADER Sentiment:** {vader_sentiment}")
                    st.write(f"**VADER Score:** {vader_score:.2f}")
                except Exception as e:
                    st.error(f"❌ Error in VADER analysis: {e}")
                
                st.subheader("Emotion Detection")
                try:
                    emotion = detect_emotion(user_review)
                    st.write(f"**Emotion:** {emotion}")
                except Exception as e:
                    st.error(f"❌ Error in emotion detection: {e}")
            
            with col2:
                st.subheader("AI Analysis")
                if st.button("Get Gemini Explanation"):
                    try:
                        with st.spinner("Analyzing with Gemini..."):
                            explanation = get_sentiment_explanation(sentiment, confidence, user_review)
                        st.write(explanation)
                    except Exception as e:
                        st.error(f"❌ Error getting Gemini explanation: {e}")
                
                if st.button("Detailed Review Analysis"):
                    try:
                        with st.spinner("Getting detailed analysis..."):
                            detailed_analysis = analyze_review_with_gemini(user_review)
                        st.write(detailed_analysis)
                    except Exception as e:
                        st.error(f"❌ Error getting detailed analysis: {e}")

    elif PAGES[page] == "gemini":
        st.title("🤖 Gemini AI Insights")
        st.write("Ask Gemini for deeper analysis or insights about movie reviews.")
        
        # Predefined questions
        st.subheader("Quick Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Analyze Review Patterns"):
                question = "What are common patterns in positive vs negative movie reviews? What makes a review convincing?"
                
        with col2:
            if st.button("🎭 Movie Preferences"):
                question = "What types of movies do audiences generally prefer based on review sentiment?"
                
        with col3:
            if st.button("📝 Writing Styles"):
                question = "How do positive and negative reviews differ in their writing style and language use?"
        
        # Custom question input
        st.subheader("Ask Your Own Question")
        custom_question = st.text_area(
            "Ask Gemini anything about movie reviews or sentiment analysis:",
            placeholder="e.g., Why are movie reviews important for the film industry?"
        )
        
        # Process questions
        question_to_process = None
        if 'question' in locals():
            question_to_process = question
        elif custom_question:
            question_to_process = custom_question
            
        if question_to_process:
            with st.spinner("🤔 Gemini is thinking..."):
                response = get_gemini_insight(question_to_process)
            
            st.subheader("💡 Gemini Response")
            st.write(response)
            
        # Tips section
        with st.expander("💡 Tips for better questions"):
            st.write("""
            - Ask about sentiment analysis techniques and best practices
            - Inquire about movie industry trends based on reviews
            - Request analysis of specific review patterns or language
            - Ask for explanations of sentiment analysis results
            - Request insights about audience preferences
            """)
        

if __name__ == "__main__":
    main()