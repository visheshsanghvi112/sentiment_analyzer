import streamlit as st
import pandas as pd
import sys
import traceback

# Production-ready app with graceful fallbacks
def main():
    st.set_page_config(
        page_title="IMDB Sentiment Analyzer", 
        layout="wide",
        page_icon="🎬"
    )
    
    st.title("🎬 IMDB Sentiment Analysis App")
    
    # Try to import and load components
    try:
        # Test datasets import
        with st.spinner("🔄 Loading datasets library..."):
            from datasets import load_dataset
        st.success("✅ Datasets library loaded successfully")
        
        # Load dataset
        with st.spinner("📊 Loading IMDB dataset (first time may take a while)..."):
            dataset = load_dataset("imdb")
            df = pd.DataFrame(dataset["train"])
        st.success(f"✅ IMDB dataset loaded: {len(df):,} reviews")
        
        # Import sentiment utils
        try:
            from sentiment_utils import analyze_sentiment, get_wordcloud, get_sentiment_distribution, detect_emotion
            st.success("✅ Sentiment analysis modules loaded")
        except Exception as e:
            st.error(f"❌ Error loading sentiment utils: {e}")
            return
        
        # Import Gemini API
        try:
            from gemini_api import get_gemini_insight, analyze_review_with_gemini, get_sentiment_explanation
            st.success("✅ Gemini API modules loaded")
        except Exception as e:
            st.warning(f"⚠️ Gemini API not available: {e}")
            
        # Show the main app
        show_main_app(df)
        
    except Exception as e:
        st.error(f"❌ Critical error during startup: {e}")
        st.code(traceback.format_exc())
        st.info("💡 This might be due to dependency conflicts. The app should work fine on Streamlit Cloud.")
        
        # Show fallback demo
        show_demo_mode()

def show_main_app(df):
    """Show the main application"""
    st.sidebar.title("🎬 Navigation")
    
    PAGES = {
        "📊 Dataset Explorer": "explorer",
        "🧠 Analyze Review": "analyze", 
        "🤖 Gemini Insights": "gemini"
    }
    
    page = st.sidebar.selectbox("Select a page", list(PAGES.keys()))
    
    # Add dataset info
    st.sidebar.metric("Total Reviews", f"{len(df):,}")
    positive_count = (df['label'] == 1).sum()
    st.sidebar.metric("Positive Reviews", f"{positive_count:,}")
    st.sidebar.metric("Negative Reviews", f"{len(df) - positive_count:,}")
    
    if PAGES[page] == "explorer":
        show_explorer_page(df)
    elif PAGES[page] == "analyze":
        show_analyzer_page()
    elif PAGES[page] == "gemini":
        show_gemini_page()

def show_explorer_page(df):
    """Dataset Explorer Page"""
    st.title("📊 IMDB Dataset Explorer")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        movie_name = st.text_input("Filter by movie name (optional):")
    with col2:
        limit = st.slider("Number of reviews", 10, 100, 30)
    
    # Filter reviews
    if movie_name:
        reviews = df[df["text"].str.contains(movie_name, case=False, na=False)]
        if len(reviews) == 0:
            st.warning(f"No reviews found for '{movie_name}'. Showing random reviews.")
            reviews = df.sample(limit)
        else:
            st.success(f"Found {len(reviews)} reviews mentioning '{movie_name}'")
            reviews = reviews.head(limit)
    else:
        reviews = df.sample(limit)
    
    # Display reviews
    st.subheader("📝 Reviews")
    for idx, (_, row) in enumerate(reviews.iterrows()):
        sentiment = "Positive 😊" if row['label'] == 1 else "Negative 😞"
        with st.expander(f"Review {idx+1} - {sentiment}"):
            st.write(row['text'][:500] + "..." if len(row['text']) > 500 else row['text'])
    
    # Show visualizations if possible
    try:
        from sentiment_utils import get_sentiment_distribution, get_wordcloud
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Sentiment Distribution")
            fig = get_sentiment_distribution(reviews)
            st.pyplot(fig)
        
        with col2:
            st.subheader("☁️ Word Cloud")
            wc_fig = get_wordcloud(reviews)
            st.pyplot(wc_fig)
            
    except Exception as e:
        st.warning(f"⚠️ Visualizations not available: {e}")

def show_analyzer_page():
    """Review Analyzer Page"""
    st.title("🧠 Analyze Your Own Review")
    
    user_review = st.text_area("Enter your movie review:", height=150, 
                              placeholder="Write your movie review here...")
    
    if user_review:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Sentiment Analysis")
            try:
                from sentiment_utils import analyze_sentiment, detect_emotion
                
                # BERT Analysis
                sentiment, confidence = analyze_sentiment(user_review)
                st.metric("BERT Sentiment", sentiment, f"{confidence:.2%} confidence")
                
                # VADER Analysis  
                vader_sentiment, vader_score = analyze_sentiment(user_review, method="vader")
                st.metric("VADER Sentiment", vader_sentiment, f"{vader_score:.2f} score")
                
                # Emotion Detection
                emotion = detect_emotion(user_review)
                st.metric("Detected Emotion", emotion)
                
            except Exception as e:
                st.error(f"❌ Sentiment analysis error: {e}")
        
        with col2:
            st.subheader("🤖 AI Insights")
            
            try:
                from gemini_api import get_sentiment_explanation, analyze_review_with_gemini
                
                if st.button("Get AI Explanation", type="primary"):
                    with st.spinner("🤔 Analyzing with AI..."):
                        explanation = get_sentiment_explanation(sentiment, confidence, user_review)
                    st.write(explanation)
                
                if st.button("Detailed Analysis"):
                    with st.spinner("🔍 Getting detailed analysis..."):
                        analysis = analyze_review_with_gemini(user_review)
                    st.write(analysis)
                    
            except Exception as e:
                st.warning(f"⚠️ AI features not available: {e}")

def show_gemini_page():
    """Gemini Insights Page"""
    st.title("🤖 Gemini AI Insights")
    
    try:
        from gemini_api import get_gemini_insight
        
        # Quick questions
        st.subheader("⚡ Quick Questions")
        col1, col2, col3 = st.columns(3)
        
        questions = [
            ("📊 Review Patterns", "What patterns distinguish positive from negative movie reviews?"),
            ("🎭 Movie Preferences", "What types of movies do audiences prefer based on sentiment?"),
            ("📝 Writing Styles", "How do positive and negative reviews differ in language use?")
        ]
        
        selected_question = None
        for i, (title, question) in enumerate(questions):
            with [col1, col2, col3][i]:
                if st.button(title):
                    selected_question = question
        
        # Custom question
        st.subheader("💭 Ask Your Own Question")
        custom_question = st.text_area("Ask anything about movie reviews or sentiment analysis:",
                                     placeholder="e.g., Why are movie reviews important for the film industry?")
        
        # Process question
        question_to_ask = selected_question or custom_question
        
        if question_to_ask:
            with st.spinner("🤔 Gemini is thinking..."):
                response = get_gemini_insight(question_to_ask)
            
            st.subheader("💡 AI Response")
            st.write(response)
            
    except Exception as e:
        st.error(f"❌ Gemini AI not available: {e}")
        st.info("💡 Make sure to add your GEMINI_API_KEY in the secrets.")

def show_demo_mode():
    """Fallback demo mode"""
    st.warning("⚠️ Running in Demo Mode due to dependency issues")
    
    st.subheader("🎬 About This App")
    st.write("""
    This is an IMDB Sentiment Analysis application with the following features:
    
    📊 **Dataset Explorer**: Browse 50,000 IMDB movie reviews
    🧠 **Sentiment Analysis**: BERT and VADER-based sentiment analysis
    🎨 **Visualizations**: Word clouds and sentiment distributions
    🤖 **AI Insights**: Gemini-powered explanations and analysis
    """)
    
    st.subheader("📝 Demo Review Analysis")
    demo_review = st.text_area("Try analyzing a review:", 
                              value="This movie was absolutely fantastic! The acting was superb and the plot was engaging from start to finish.")
    
    if demo_review:
        st.write("**Demo Analysis:**")
        st.write("- **Sentiment**: Positive")
        st.write("- **Confidence**: 95%")
        st.write("- **Emotion**: Joy")
        st.info("💡 Full analysis available when dependencies are properly installed.")

if __name__ == "__main__":
    main()
