"""
Simple IMDB Movie Sentiment Analyzer
A clean, straightforward Streamlit app for movie review sentiment analysis
"""

import streamlit as st
import pandas as pd
import logging
from logger_config import setup_logging
from sentiment_utils import SentimentAnalyzer
from utils import clean_text, load_imdb_sample, format_confidence, get_sentiment_color
from config import Config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_analyzer():
    """Load sentiment analyzer (cached)"""
    return SentimentAnalyzer()

@st.cache_data
def load_sample_data(num_samples=250):
    """Load sample IMDB data (cached)"""
    return load_imdb_sample(num_samples)

def main():
    """Main app function"""
    
    st.title("🎬 IMDB Movie Sentiment Analyzer")
    st.markdown("Analyze the sentiment of movie reviews using machine learning")
    
    # Load analyzer
    with st.spinner("Loading AI model..."):
        analyzer = load_analyzer()
    
    if not analyzer.is_available():
        st.error("❌ Failed to load sentiment analysis model")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📝 Analyze Text", "📊 Sample Data"])
    
    with tab1:
        analyze_text_tab(analyzer)
    
    with tab2:
        sample_data_tab(analyzer)

def analyze_text_tab(analyzer):
    """Text analysis tab"""
    
    st.header("Analyze Your Text")
    
    # Text input
    text_input = st.text_area(
        "Enter movie review text:",
        placeholder="Type or paste a movie review here...",
        height=150
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    if analyze_button and text_input.strip():
        with st.spinner("Analyzing sentiment..."):
            # Clean and analyze text
            cleaned_text = clean_text(text_input)
            result = analyzer.analyze(cleaned_text)
            
            # Display results
            if 'error' not in result:
                sentiment = result['sentiment']
                confidence = result['confidence']
                
                # Create result display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Sentiment",
                        value=sentiment.title()
                    )
                
                with col2:
                    st.metric(
                        label="Confidence",
                        value=format_confidence(confidence)
                    )
                
                with col3:
                    st.metric(
                        label="Text Length",
                        value=f"{result.get('text_length', 0)} chars"
                    )
                
                # Sentiment indicator
                color = get_sentiment_color(sentiment)
                st.markdown(
                    f"""
                    <div style="
                        padding: 1rem;
                        border-radius: 0.5rem;
                        background-color: {color}20;
                        border-left: 4px solid {color};
                        margin: 1rem 0;
                    ">
                        <h4 style="color: {color}; margin: 0;">
                            {sentiment.title()} Sentiment ({format_confidence(confidence)})
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            else:
                st.error(f"❌ Error: {result['error']}")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze")

def sample_data_tab(analyzer):
    """Sample data analysis tab"""
    
    st.header("IMDB Sample Analysis")
    
    # Sample size selector
    sample_size = st.slider(
        "Number of reviews to analyze:",
        min_value=50,
        max_value=500,
        value=250,
        step=25,
        help="More reviews = more accurate results, but slower processing"
    )
    
    # Load sample data
    with st.spinner(f"Loading {sample_size} IMDB sample reviews..."):
        sample_data = load_sample_data(sample_size)
    
    if not sample_data:
        st.error("❌ Failed to load sample data")
        return
    
    actual_count = len(sample_data)
    if actual_count < sample_size:
        st.warning(f"⚠️ Only {actual_count} reviews available (requested {sample_size})")
    else:
        st.success(f"✅ Loaded {actual_count} IMDB reviews")
    
    # Analyze samples button
    if st.button("🔍 Analyze Sample Reviews", type="primary"):
        
        progress_bar = st.progress(0)
        results = []
        
        for i, sample in enumerate(sample_data):
            # Update progress
            progress_bar.progress((i + 1) / len(sample_data))
            
            # Analyze sentiment
            result = analyzer.analyze(sample['text'])
            
            results.append({
                'Review': sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text'],
                'Actual': sample['label'],
                'Predicted': result.get('sentiment', 'unknown'),
                'Confidence': result.get('confidence', 0.0),
                'Match': sample['label'] == result.get('sentiment', 'unknown')
            })
        
        # Create results dataframe
        df = pd.DataFrame(results)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = df['Match'].mean() * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        with col2:
            avg_confidence = df['Confidence'].mean() * 100
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col3:
            total_samples = len(df)
            st.metric("Total Samples", total_samples)
        
        # Show sample reviews for understanding
        st.subheader("📝 Sample Reviews (First 10)")
        st.markdown("*Here are some examples so you can see how the model is performing:*")
        
        sample_df = df.head(10).copy()
        
        # Create a more readable display for samples
        for idx, row in sample_df.iterrows():
            match_icon = "✅" if row['Match'] else "❌"
            confidence_color = "green" if row['Confidence'] > 0.7 else "orange" if row['Confidence'] > 0.5 else "red"
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{match_icon} Review {idx + 1}** ({row['Dataset']})")
                    st.markdown(f"*\"{row['Review']}\"*")
                    st.markdown(f"**Actual:** {row['Actual']} | **Predicted:** {row['Predicted']}")
                
                with col2:
                    st.markdown(f"**Confidence**")
                    st.markdown(f":{confidence_color}[{row['Confidence']:.1%}]")
                
                st.divider()
        
        # Display full results table
        st.subheader("📊 Complete Results Table")
        
        # Color code the dataframe
        def color_matches(val):
            if val == True:
                return 'background-color: #d4edda'
            elif val == False:
                return 'background-color: #f8d7da'
            return ''
        
        styled_df = df.style.map(color_matches, subset=['Match'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary
        correct = df['Match'].sum()
        total = len(df)
        st.info(f"📊 **Summary**: {correct}/{total} predictions were correct ({accuracy:.1f}% accuracy)")

if __name__ == "__main__":
    main()