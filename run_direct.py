"""
Direct Streamlit launcher - alternative method
"""

import os
import sys

def main():
    """Launch Streamlit directly"""
    
    print("🚀 Starting IMDB Sentiment Analyzer (Direct Mode)...")
    print("📱 App will be available at: http://localhost:8501")
    print()
    print("💡 If you see JavaScript errors:")
    print("   1. Clear browser cache (Ctrl+Shift+Delete)")
    print("   2. Try incognito/private browsing mode") 
    print("   3. Try a different browser")
    print("   4. Restart your browser completely")
    print()
    
    # Change to app directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run streamlit directly
    os.system(f"{sys.executable} -m streamlit run app.py --server.port 8501")

if __name__ == "__main__":
    main()