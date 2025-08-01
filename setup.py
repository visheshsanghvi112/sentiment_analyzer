#!/usr/bin/env python3
"""
Setup script for IMDB Sentiment Analysis App
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
        return False
    return True

def setup_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        print("🔐 Setting up environment file...")
        with open('.env', 'w') as f:
            f.write("# Add your Gemini API key here\n")
            f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
        print("✅ Created .env file. Please add your Gemini API key.")
        print("   Get your key from: https://makersuite.google.com/app/apikey")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🎬 IMDB Sentiment Analysis App Setup")
    print("="*40)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Setup environment
    setup_env_file()
    
    print("\n🚀 Setup complete!")
    print("\nNext steps:")
    print("1. Add your Gemini API key to the .env file")
    print("2. Run: streamlit run app.py")
    print("\nEnjoy analyzing movie reviews! 🎭")

if __name__ == "__main__":
    main()
