import streamlit as st
import sys
import os

# Add error handling for imports
def safe_import():
    try:
        import pandas as pd
        return pd, True, ""
    except Exception as e:
        return None, False, str(e)

def main():
    st.set_page_config(
        page_title="IMDB Sentiment Analyzer - Diagnosis", 
        layout="wide",
        page_icon="🎬"
    )
    
    st.title("🔧 Environment Diagnosis")
    
    # Check Python version
    st.subheader("Python Environment")
    st.write(f"**Python Version:** {sys.version}")
    st.write(f"**Python Executable:** {sys.executable}")
    
    # Test imports one by one
    st.subheader("Package Import Tests")
    
    # Test basic packages
    try:
        import numpy as np
        st.success(f"✅ NumPy: {np.__version__}")
    except Exception as e:
        st.error(f"❌ NumPy: {e}")
    
    # Test pandas
    pandas, success, error = safe_import()
    if success:
        st.success(f"✅ Pandas: {pandas.__version__}")
    else:
        st.error(f"❌ Pandas: {error}")
    
    # Test other packages
    packages_to_test = [
        ("streamlit", "st"),
        ("matplotlib", "matplotlib"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("nltk", "nltk"),
    ]
    
    for package_name, import_name in packages_to_test:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'Unknown')
            st.success(f"✅ {package_name}: {version}")
        except Exception as e:
            st.error(f"❌ {package_name}: {e}")
    
    # Show environment info
    st.subheader("Environment Details")
    st.write("**PATH Variables:**")
    for path in sys.path[:5]:  # Show first 5 paths
        st.code(path)
    
    # Show installed packages
    if st.button("Show All Installed Packages"):
        try:
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                  capture_output=True, text=True)
            st.code(result.stdout)
        except Exception as e:
            st.error(f"Error getting package list: {e}")

if __name__ == "__main__":
    main()
