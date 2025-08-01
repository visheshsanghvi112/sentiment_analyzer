@echo off
echo 🎬 IMDB Sentiment Analysis App - Windows Setup
echo ==========================================

echo.
echo 📦 Installing Python packages...
pip install -r requirements.txt

echo.
echo 🔐 Setting up environment...
if not exist .env (
    echo # Add your Gemini API key here > .env
    echo GEMINI_API_KEY=your_gemini_api_key_here >> .env
    echo ✅ Created .env file
    echo    Please add your Gemini API key from: https://makersuite.google.com/app/apikey
) else (
    echo ✅ .env file already exists
)

echo.
echo 🚀 Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file and add your Gemini API key
echo 2. Run: streamlit run app.py
echo.
echo Press any key to exit...
pause > nul
