import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    st.set_page_config(
        page_title="Sentiment Analyzer Test", 
        layout="wide",
        page_icon="🎬"
    )

    st.title("🎬 Sentiment Analyzer - Test Version")
    st.write("This is a test version to verify the app structure works.")

    # Sidebar
    st.sidebar.title("🎬 Sentiment Analyzer")
    st.sidebar.markdown("---")
    
    # Test data
    test_data = {
        'text': [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible movie, waste of time and money.",
            "It was okay, nothing special but not bad either."
        ],
        'label': [1, 0, 0]  # 1 for positive, 0 for negative
    }
    
    df = pd.DataFrame(test_data)
    
    st.sidebar.metric("Total Reviews", len(df))
    st.sidebar.metric("Positive Reviews", (df['label'] == 1).sum())
    st.sidebar.metric("Negative Reviews", (df['label'] == 0).sum())
    
    st.sidebar.markdown("---")
    
    # Main content
    st.header("📊 Test Dataset")
    st.write("Sample reviews for testing:")
    
    # Display the test data
    st.dataframe(df, use_container_width=True)
    
    # Test sentiment analysis
    st.header("🧠 Simple Sentiment Analysis")
    
    user_text = st.text_area("Enter a review to analyze:", 
                            placeholder="Type your review here...",
                            height=100)
    
    if st.button("Analyze Sentiment"):
        if user_text:
            # Simple sentiment analysis based on keywords
            positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'love', 'wonderful']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointing', 'waste', 'boring']
            
            text_lower = user_text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "POSITIVE"
                confidence = min(0.9, 0.5 + (positive_count * 0.1))
            elif negative_count > positive_count:
                sentiment = "NEGATIVE"
                confidence = min(0.9, 0.5 + (negative_count * 0.1))
            else:
                sentiment = "NEUTRAL"
                confidence = 0.5
            
            st.success(f"✅ Sentiment: {sentiment}")
            st.info(f"📊 Confidence: {confidence:.2f}")
            
            # Show analysis
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positive Words Found", positive_count)
            with col2:
                st.metric("Negative Words Found", negative_count)
        else:
            st.warning("Please enter some text to analyze.")
    
    # Test visualization
    st.header("📈 Test Visualizations")
    
    # Create a simple chart
    sentiment_counts = df['label'].value_counts()
    st.bar_chart(sentiment_counts)
    
    st.success("✅ App structure is working correctly!")
    st.info("This test version confirms the basic Streamlit app structure works.")

if __name__ == "__main__":
    main() 