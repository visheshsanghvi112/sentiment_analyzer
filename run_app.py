"""
Simple app launcher for IMDB Sentiment Analyzer
"""

import os
import sys
import logging
import subprocess
from logger_config import setup_logging

def main():
    """Launch the Streamlit app"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting IMDB Sentiment Analyzer...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        logger.error("❌ app.py not found. Make sure you're in the correct directory.")
        return
    
    # Check basic imports
    try:
        import streamlit
        import transformers
        import torch
        logger.info("All required packages are available")
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        logger.info("Run: pip install -r requirements.txt")
        return
    
    # Launch Streamlit
    try:
        logger.info("Launching Streamlit app...")
        logger.info("App will be available at: http://localhost:8501")
        logger.info("If you see JavaScript errors, try:")
        logger.info("1. Clear browser cache (Ctrl+Shift+Delete)")
        logger.info("2. Try incognito/private browsing mode")
        logger.info("3. Try a different browser")
        
        # Run streamlit with cache clearing
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        logger.info("App stopped by user")
    except Exception as e:
        logger.error(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main()