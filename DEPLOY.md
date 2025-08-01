# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy Steps

### 1. Upload to GitHub
1. Create a new GitHub repository
2. Upload all files EXCEPT any `.env` files with real API keys
3. Make sure `.env.example` doesn't contain your real API key

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set main file path: `app.py`
5. Click "Deploy"

### 3. Add API Key (CRITICAL)
1. Go to your app settings (gear icon)
2. Click on "Secrets" tab
3. Add this content:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
4. Click "Save"

### 4. Wait for Deployment
- First deployment takes 3-5 minutes
- The app will download the IMDB dataset (may take extra time)
- Once deployed, it will be available at your Streamlit URL

## Important Notes

### ⚠️ Security
- **NEVER** commit real API keys to GitHub
- Always use Streamlit Cloud's Secrets feature for API keys
- The `.env.example` file should only contain dummy values

### 🔧 Troubleshooting

**App won't start?**
- Check the logs in Streamlit Cloud
- Ensure all dependencies are in `requirements.txt`
- Make sure API key is correctly set in Secrets

**Models taking too long to load?**
- First run always takes longer (models are downloaded)
- Subsequent runs will be faster due to caching

**Dataset not loading?**
- Check internet connection
- Hugging Face servers might be busy - try refreshing

**Gemini not working?**
- Verify API key is correct in Secrets
- Check if you have API quota remaining
- Make sure key has proper permissions

### 📱 Features That Work
- ✅ Dataset exploration with 50K reviews
- ✅ BERT sentiment analysis
- ✅ VADER sentiment scoring
- ✅ Word clouds and visualizations
- ✅ Emotion detection
- ✅ Gemini AI insights (if API key provided)
- ✅ Responsive design for mobile

### 🎯 Performance Tips
- Models are cached after first load
- Dataset is cached for better performance
- Use reasonable limits for review exploration
- Gemini responses are not cached (for freshness)

## File Structure for Deployment
```
your-repo/
├── app.py                 # Main app (required)
├── sentiment_utils.py     # Sentiment functions (required)
├── gemini_api.py         # Gemini integration (required)
├── requirements.txt      # Dependencies (required)
├── README.md            # Documentation
├── .streamlit/
│   ├── config.toml      # App configuration
│   └── secrets.toml.example  # Secret template
└── .env.example         # Environment template
```

## Ready to Deploy? 🚀
Your app is production-ready with:
- ✅ Error handling for all major functions
- ✅ User-friendly error messages
- ✅ Proper caching for performance
- ✅ Mobile-responsive design
- ✅ Secure API key handling
- ✅ Professional UI/UX

Happy deploying! 🎬
