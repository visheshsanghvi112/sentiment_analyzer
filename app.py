from datasets import load_dataset
import pandas as pd
import streamlit as st
import sys
import traceback
import time
from typing import List, Dict, Tuple, Optional

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

# Initialize variables for modules
MovieSearchEngine = None
EmbeddingEngine = None
TextProcessor = None
MovieRecommender = None
ADVANCED_FEATURES = False

# Import basic modules (advanced modules were removed)
def import_modules():
    global MovieSearchEngine, EmbeddingEngine, TextProcessor, MovieRecommender, ADVANCED_FEATURES
    
    try:
        from search_utils_basic import MovieSearchEngine, TFIDFEmbeddingEngine as EmbeddingEngine, TextProcessor
        from recommender_basic import MovieRecommender
        ADVANCED_FEATURES = False
        st.success("✅ TF-IDF features loaded successfully")
        return True
    except ImportError as e:
        st.error(f"❌ Error importing modules: {e}")
        return False

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

# Initialize dataset and engines
def initialize_app():
    global df, search_engine, recommender, text_processor
    
    # Import modules first
    if not import_modules():
        st.stop()
    
    try:
        df = load_imdb_data()
        search_engine = MovieSearchEngine(df)
        recommender = MovieRecommender(df)
        text_processor = TextProcessor()
        return True
    except Exception as e:
        st.error(f"❌ Critical error: {e}")
        return False

def main():
    st.set_page_config(
        page_title="🎬 AI Movie Sentiment Analyzer", 
        layout="wide",
        page_icon="🎬"
    )
    
    # Initialize the app
    if not initialize_app():
        st.stop()

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .recommendation-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI Movie Sentiment Analyzer</h1>
        <p>Discover movies, analyze reviews, and get intelligent recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    # Show feature status
    st.success("ℹ️ Running with TF-IDF embeddings for fast and reliable recommendations.")

    # Sidebar
    st.sidebar.title("🎬 Navigation")
    st.sidebar.markdown("---")
    
    # Dataset info
    if not df.empty:
        st.sidebar.markdown("### 📊 Dataset Stats")
        st.sidebar.metric("Total Reviews", f"{len(df):,}")
        positive_count = (df['label'] == 1).sum()
        st.sidebar.metric("Positive Reviews", f"{positive_count:,}")
        st.sidebar.metric("Negative Reviews", f"{len(df) - positive_count:,}")
    
    st.sidebar.markdown("---")

    PAGES = {
        "🔍 Smart Movie Search": "search",
        "📊 Review Analysis": "analysis", 
        "🎯 Movie Recommendations": "recommendations",
        "💭 Mood-Based Search": "mood_search",
        "🤖 AI Insights": "ai_insights",
        "📈 Dataset Explorer": "explorer"
    }

    page = st.sidebar.selectbox("Select a page", list(PAGES.keys()))
    
    # Add helpful information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Features")
    st.sidebar.markdown("""
    - **Smart Search** with autocomplete
    - **Fuzzy Matching** for typos
    - **AI Recommendations** with explanations
    - **Mood-Based Search**
    - **Real-time Analysis**
    - **Gemini AI Insights**
    """)
    
    # Handle empty dataset
    if df.empty:
        st.error("Failed to load IMDB dataset. Please check your internet connection.")
        return

    if PAGES[page] == "search":
        render_smart_search_page()
    elif PAGES[page] == "analysis":
        render_analysis_page()
    elif PAGES[page] == "recommendations":
        render_recommendations_page()
    elif PAGES[page] == "mood_search":
        render_mood_search_page()
    elif PAGES[page] == "ai_insights":
        render_ai_insights_page()
    elif PAGES[page] == "explorer":
        render_explorer_page()

def render_smart_search_page():
    """Smart movie search with autocomplete and fuzzy matching"""
    st.title("🔍 Smart Movie Search")
    st.write("Search for movies with intelligent autocomplete and fuzzy matching.")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for a movie:",
            placeholder="Start typing... (e.g., 'titanic', 'godfather')",
            key="movie_search"
        )
    
    with col2:
        search_mode = st.selectbox(
            "Search Mode",
            ["Autocomplete", "Fuzzy Search"],
            key="search_mode"
        )
    
    # Handle search
    if search_query:
        if search_mode == "Autocomplete":
            suggestions = search_engine.autocomplete(search_query, limit=10)
            
            if suggestions:
                st.success(f"Found {len(suggestions)} suggestions")
                
                # Show suggestions in a selectbox
                selected_movie = st.selectbox(
                    "Select a movie:",
                    suggestions,
                    key="movie_suggestions"
                )
                
                if selected_movie:
                    display_movie_analysis(selected_movie)
            else:
                st.warning("No suggestions found. Try a different search term.")
        
        else:  # Fuzzy Search
            fuzzy_matches = search_engine.fuzzy_search(search_query, threshold=60)
            
            if fuzzy_matches:
                st.success(f"Found {len(fuzzy_matches)} fuzzy matches")
                
                # Display matches with similarity scores
                for movie, score in fuzzy_matches:
                    with st.expander(f"{movie} (similarity: {score}%)"):
                        if st.button(f"Analyze {movie}", key=f"analyze_{movie}"):
                            display_movie_analysis(movie)
            else:
                st.warning("No fuzzy matches found. Try a different search term.")

def display_movie_analysis(movie_name: str):
    """Display comprehensive analysis for a selected movie"""
    st.markdown(f"## 🎬 Analysis for: {movie_name}")
    
    # Get movie data
    movie_reviews = search_engine.get_movie_reviews(movie_name, limit=50)
    movie_stats = search_engine.get_movie_stats(movie_name)
    
    if movie_reviews.empty:
        st.error(f"No reviews found for '{movie_name}'")
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", movie_stats['total_reviews'])
    
    with col2:
        st.metric("Positive Reviews", movie_stats['positive_reviews'])
    
    with col3:
        st.metric("Negative Reviews", movie_stats['negative_reviews'])
    
    with col4:
        st.metric("Positive Ratio", f"{movie_stats['positive_ratio']:.1%}")
    
    # Sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        try:
            fig = get_sentiment_distribution(movie_reviews)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating sentiment chart: {e}")
    
    with col2:
        st.subheader("☁️ Word Cloud")
        try:
            wc_fig = get_wordcloud(movie_reviews)
            st.pyplot(wc_fig)
        except Exception as e:
            st.error(f"Error creating word cloud: {e}")
    
    # Top reviews
    st.subheader("📝 Top Reviews")
    
    # Get top positive and negative reviews
    positive_reviews = movie_reviews[movie_reviews['label'] == 1].head(3)
    negative_reviews = movie_reviews[movie_reviews['label'] == 0].head(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👍 Positive Reviews**")
        for idx, (_, review) in enumerate(positive_reviews.iterrows()):
            with st.expander(f"Positive Review {idx+1}"):
                st.write(review['text'][:300] + "..." if len(review['text']) > 300 else review['text'])
    
    with col2:
        st.markdown("**👎 Negative Reviews**")
        for idx, (_, review) in enumerate(negative_reviews.iterrows()):
            with st.expander(f"Negative Review {idx+1}"):
                st.write(review['text'][:300] + "..." if len(review['text']) > 300 else review['text'])
    
    # AI Analysis button
    if st.button("🤖 Get AI Analysis", key="ai_analysis"):
        with st.spinner("Analyzing with AI..."):
            try:
                # Simple analysis without advanced Gemini features
                st.subheader("🧠 AI Analysis")
                st.write(f"Based on the analysis of {movie_stats['total_reviews']} reviews for '{movie_name}':")
                st.write(f"- **Overall Sentiment**: {'Positive' if movie_stats['positive_ratio'] > 0.5 else 'Negative'}")
                st.write(f"- **Audience Reception**: {movie_stats['positive_ratio']:.1%} positive reviews")
                st.write(f"- **Review Volume**: {movie_stats['total_reviews']} total reviews")
                
                if movie_stats['positive_ratio'] > 0.7:
                    st.success("This movie is well-received by audiences!")
                elif movie_stats['positive_ratio'] < 0.3:
                    st.error("This movie has mixed to negative reception.")
                else:
                    st.info("This movie has mixed reception from audiences.")
                    
            except Exception as e:
                st.error(f"Error getting AI analysis: {e}")

def render_analysis_page():
    """Review analysis page"""
    st.title("📊 Review Analysis")
    st.write("Analyze your own movie reviews with advanced sentiment analysis.")
    
    # Review input
    user_review = st.text_area(
        "Enter your movie review:",
        height=150,
        placeholder="Write your review here..."
    )
    
    if user_review:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Sentiment Analysis")
            try:
                sentiment, confidence = analyze_sentiment(user_review)
                st.write(f"**Sentiment:** {sentiment}")
                st.write(f"**Confidence:** {confidence:.2f}")
                
                # Display sentiment with color
                if sentiment == "POSITIVE":
                    st.success("✅ Positive")
                elif sentiment == "NEGATIVE":
                    st.error("❌ Negative")
                else:
                    st.info("➖ Neutral")
                    
            except Exception as e:
                st.error(f"❌ Error in sentiment analysis: {e}")
            
            st.subheader("📊 VADER Score")
            try:
                vader_sentiment, vader_score = analyze_sentiment(user_review, method="vader")
                st.write(f"**VADER Sentiment:** {vader_sentiment}")
                st.write(f"**VADER Score:** {vader_score:.2f}")
            except Exception as e:
                st.error(f"❌ Error in VADER analysis: {e}")
            
            st.subheader("😊 Emotion Detection")
            try:
                emotion = detect_emotion(user_review)
                st.write(f"**Detected Emotion:** {emotion}")
            except Exception as e:
                st.error(f"❌ Error in emotion detection: {e}")
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
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

def render_recommendations_page():
    """Movie recommendations page"""
    st.title("🎯 Movie Recommendations")
    st.write("Get intelligent movie recommendations based on your preferences.")
    
    # Recommendation type selection
    rec_type = st.selectbox(
        "Choose recommendation type:",
        ["Based on Movie", "Based on Review", "Movie Clusters"]
    )
    
    if rec_type == "Based on Movie":
        # Movie-based recommendations
        st.subheader("🎬 Movie-Based Recommendations")
        
        # Movie search with autocomplete
        movie_query = st.text_input(
            "Enter a movie you like:",
            placeholder="e.g., Titanic, The Godfather"
        )
        
        if movie_query:
            suggestions = search_engine.autocomplete(movie_query, limit=5)
            if suggestions:
                selected_movie = st.selectbox("Select movie:", suggestions)
                
                if selected_movie and st.button("Get Recommendations"):
                    with st.spinner("Finding similar movies..."):
                        recommendations = recommender.find_similar_movies(selected_movie, top_k=5)
                        
                        if recommendations:
                            st.success(f"Found {len(recommendations)} similar movies")
                            
                            for i, (movie, similarity) in enumerate(recommendations):
                                with st.expander(f"#{i+1}: {movie} (similarity: {similarity:.2f})"):
                                    # Get movie features for explanation
                                    source_features = recommender.get_movie_features(selected_movie)
                                    target_features = recommender.get_movie_features(movie)
                                    
                                    # Get AI explanation
                                    try:
                                        explanation = recommender.explain_recommendation(selected_movie, movie)
                                        st.write(explanation)
                                    except Exception as e:
                                        st.error(f"Error getting explanation: {e}")
                        else:
                            st.warning("No similar movies found.")
    
    elif rec_type == "Based on Review":
        # Review-based recommendations
        st.subheader("📝 Review-Based Recommendations")
        
        review_text = st.text_area(
            "Enter a review or description:",
            placeholder="Describe what you're looking for in a movie..."
        )
        
        if review_text and st.button("Get Recommendations"):
            with st.spinner("Finding movies based on your review..."):
                recommendations = recommender.get_review_based_recommendations(review_text, top_k=5)
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} matching movies")
                    
                    for i, (movie, similarity) in enumerate(recommendations):
                        with st.expander(f"#{i+1}: {movie} (similarity: {similarity:.2f})"):
                            # Get movie features
                            movie_features = recommender.get_movie_features(movie)
                            if movie_features:
                                st.write(f"**Keywords:** {', '.join(movie_features.get('keywords', [])[:5])}")
                                st.write(f"**Sentiment:** {movie_features.get('avg_sentiment', 0):.2f}")
                                st.write(f"**Positive Ratio:** {movie_features.get('positive_ratio', 0):.1%}")
                else:
                    st.warning("No matching movies found.")
    
    else:  # Movie Clusters
        st.subheader("📊 Movie Clusters")
        st.write("Discover movies grouped by similar themes and characteristics.")
        
        if st.button("Generate Movie Clusters"):
            with st.spinner("Creating movie clusters..."):
                clusters = recommender.get_movie_clusters(n_clusters=5)
                
                if clusters:
                    for cluster_id, movies in clusters.items():
                        with st.expander(f"Cluster {cluster_id + 1} ({len(movies)} movies)"):
                            st.write(", ".join(movies[:10]))
                            if len(movies) > 10:
                                st.write(f"... and {len(movies) - 10} more")
                else:
                    st.warning("Unable to generate clusters.")

def render_mood_search_page():
    """Mood-based movie search"""
    st.title("💭 Mood-Based Movie Search")
    st.write("Find movies that match your current mood or desired feeling.")
    
    # Mood input
    mood_description = st.text_input(
        "Describe your mood or desired feeling:",
        placeholder="e.g., feel-good, dark, romantic, inspiring, relaxing"
    )
    
    if mood_description and st.button("Find Movies"):
        with st.spinner("Finding movies for your mood..."):
            recommendations = recommender.get_mood_based_recommendations(mood_description, top_k=5)
            
            if recommendations:
                st.success(f"Found {len(recommendations)} movies matching your mood")
                
                # Display recommendations
                for i, (movie, similarity) in enumerate(recommendations):
                    with st.expander(f"#{i+1}: {movie} (match: {similarity:.2f})"):
                        # Get movie features
                        movie_features = recommender.get_movie_features(movie)
                        if movie_features:
                            st.write(f"**Keywords:** {', '.join(movie_features.get('keywords', [])[:5])}")
                            st.write(f"**Sentiment:** {movie_features.get('avg_sentiment', 0):.2f}")
                            st.write(f"**Positive Ratio:** {movie_features.get('positive_ratio', 0):.1%}")
                
                # Get AI analysis of mood-based recommendations
                if st.button("🤖 Get AI Analysis"):
                    with st.spinner("Analyzing recommendations..."):
                        try:
                            st.subheader("🧠 AI Analysis")
                            st.write(f"Based on your mood '{mood_description}', these movies were selected because:")
                            st.write("- They share emotional themes with your desired mood")
                            st.write("- Their review content matches your emotional preferences")
                            st.write("- They have similar audience reception patterns")
                        except Exception as e:
                            st.error(f"Error getting AI analysis: {e}")
            else:
                st.warning("No movies found for this mood.")

def render_ai_insights_page():
    """AI insights page"""
    st.title("🤖 AI Insights")
    st.write("Ask Gemini for deeper analysis and insights about movies and reviews.")
    
    # Predefined questions
    st.subheader("💡 Quick Questions")
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
    st.subheader("❓ Ask Your Own Question")
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

def render_explorer_page():
    """Dataset explorer page"""
    st.title("📈 Dataset Explorer")
    st.write("Browse and filter movie reviews from the IMDB dataset.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        movie_name = st.text_input("Filter by movie name (optional):")
    with col2:
        limit = st.slider("Number of reviews to show", 10, 100, 30)
    
    if movie_name:
        reviews = search_engine.get_movie_reviews(movie_name, limit)
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

if __name__ == "__main__":
    main() 