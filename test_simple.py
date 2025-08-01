import streamlit as st
import pandas as pd
import numpy as np

# Simple test app
def main():
    st.set_page_config(
        page_title="IMDB Sentiment Analyzer - Test", 
        layout="wide",
        page_icon="🎬"
    )
    
    st.title("🎬 IMDB Sentiment Analyzer - Test Version")
    st.write("Testing basic functionality...")
    
    # Test basic functionality
    st.subheader("✅ Streamlit Working")
    st.write("Streamlit is running successfully!")
    
    st.subheader("✅ Pandas Working")
    test_df = pd.DataFrame({'test': [1, 2, 3], 'values': ['a', 'b', 'c']})
    st.dataframe(test_df)
    
    st.subheader("✅ NumPy Working")
    test_array = np.array([1, 2, 3, 4, 5])
    st.write(f"NumPy array: {test_array}")
    
    # Test user input
    st.subheader("📝 Test Input")
    user_input = st.text_area("Enter some text:")
    if user_input:
        st.write(f"You entered: {user_input}")
        st.write(f"Length: {len(user_input)} characters")
    
    st.success("🎉 Basic app functionality is working!")

if __name__ == "__main__":
    main()
