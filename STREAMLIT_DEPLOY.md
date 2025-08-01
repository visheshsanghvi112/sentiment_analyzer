# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Step 1: Repository Setup ✅
- ✅ Code is already pushed to GitHub: https://github.com/visheshsanghvi112/sentiment_analyzer
- ✅ All required files are included
- ✅ `.env.example` file is present (no real API keys)

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Configure your app**:
   - **Repository**: `visheshsanghvi112/sentiment_analyzer`
   - **Branch**: `master`
   - **Main file path**: `app.py`
   - **Python version**: 3.9 or higher

### Step 3: Add Secrets (Optional)

If you want to use Gemini AI features:

1. **Get Gemini API Key**: https://makersuite.google.com/app/apikey
2. **In Streamlit Cloud**, go to your app settings
3. **Add secret**:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```

### Step 4: Deploy

Click **"Deploy!"** and wait for the build to complete.

## 🎯 App Features

Your deployed app will include:

- **📊 Dataset Explorer**: Browse IMDB movie reviews
- **🧠 Review Analysis**: Analyze custom reviews
- **🤖 AI Insights**: Gemini-powered analysis (if API key provided)
- **📈 Visualizations**: Charts and word clouds
- **🎨 Modern UI**: Beautiful, responsive design

## 🔧 Troubleshooting

### Common Issues:

1. **Build fails**: Check that all dependencies are in `requirements.txt`
2. **Import errors**: Make sure all Python files are in the repository
3. **API errors**: Verify your Gemini API key is correct

### Test the App:

1. **Simple test**: Run `test_app_simple.py` locally first
2. **Full test**: Run `app.py` locally to test all features
3. **Deploy**: Push to GitHub and deploy on Streamlit Cloud

## 📱 App URL

Once deployed, your app will be available at:
```
https://sentiment-analyzer-[your-username].streamlit.app
```

## 🎉 Success!

Your sentiment analyzer is now live on Streamlit Cloud! 🚀 