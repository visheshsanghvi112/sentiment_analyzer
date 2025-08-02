#!/usr/bin/env python3
"""
Simple startup script for the Sentiment Analyzer app
"""

import subprocess
import sys
import os

def main():
    print("🎬 Starting AI Movie Sentiment Analyzer...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory")
        print("💡 Please run this script from the sentiment_analyzer folder")
        return
    
    # Check if streamlit is available
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Try to run the app
    print("\n🚀 Launching the app...")
    print("💡 The app will open in your browser at http://localhost:8501")
    print("💡 Press Ctrl+C to stop the app")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")
        print("💡 Try running: streamlit run app.py")

if __name__ == "__main__":
    main() 