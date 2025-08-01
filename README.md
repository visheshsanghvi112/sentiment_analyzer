# 🎬 AI Movie Sentiment Analyzer

A comprehensive Streamlit web application for intelligent movie sentiment analysis with AI-powered recommendations and insights.

## 🚀 Enhanced Features

### 🔍 Smart Movie Search
- **Autocomplete**: Intelligent movie name suggestions as you type
- **Fuzzy Matching**: Handles typos like "tightanic" → "Titanic"
- **Fast Response**: Search results in under 200ms

### 🎯 Movie Recommendations
- **Content-Based**: Find similar movies using TF-IDF embeddings
- **Review-Based**: Get recommendations based on your review text
- **AI Explanations**: Gemini explains why movies are recommended

### 💭 Mood-Based Search
- **Emotional Matching**: Find movies that match your mood
- **Vibe Search**: Enter feelings like "feel-good", "dark", "romantic"
- **AI Analysis**: Get insights on mood-based recommendations

### 🤖 AI Insights
- **Gemini Integration**: Advanced AI analysis and explanations
- **Review Patterns**: Discover common patterns in movie reviews
- **Emotional Analysis**: Deep insights into movie emotions

### 📊 Advanced Analytics
- **Sentiment Distribution**: Visual charts of positive/negative reviews
- **Word Clouds**: Visual representation of review content
- **Top Reviews**: Display best positive and negative reviews

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/visheshsanghvi112/sentiment_analyzer.git
   cd sentiment_analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gemini API** (recommended for full features):
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create `.streamlit/secrets.toml` with:
     ```toml
     GEMINI_API_KEY = "your_actual_gemini_api_key_here"
     ```

4. **Run the enhanced app**:
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage

### 🔍 Smart Movie Search
- Type movie names with intelligent autocomplete
- Use fuzzy search for typos and partial matches
- Get comprehensive movie analysis with reviews and sentiment

### 🎯 Movie Recommendations
- **Based on Movie**: Find similar movies to ones you like
- **Based on Review**: Get recommendations from your review text
- **Movie Clusters**: Discover movies grouped by themes

### 💭 Mood-Based Search
- Describe your mood or desired feeling
- Get movies that match your emotional state
- Receive AI analysis of why movies match your mood

### 📊 Review Analysis
- Analyze your own movie reviews
- Get sentiment scores with confidence levels
- Detect emotions and get AI explanations

### 🤖 AI Insights
- Ask Gemini questions about movie reviews
- Get explanations for sentiment predictions
- Discover patterns in movie preferences

## 🛠️ Technical Architecture

### Core Components
- **`app.py`**: Main application with fallback mechanism
- **`search_utils_basic.py`**: TF-IDF based search engine
- **`recommender_basic.py`**: Content-based recommendation system
- **`sentiment_utils.py`**: Sentiment analysis with graceful fallback
- **`gemini_api.py`**: Gemini AI integration

### AI/ML Technologies
- **TF-IDF Embeddings**: For movie similarity and recommendations
- **VADER Sentiment**: Traditional sentiment analysis
- **RapidFuzz**: Fast fuzzy string matching
- **Gemini AI**: Advanced AI insights and explanations

### Performance Features
- **Caching**: Streamlit caching for better performance
- **Fallback Mechanism**: Works with or without advanced ML libraries
- **Real-time**: Fast search and analysis responses
- **Graceful Degradation**: App works in any environment

## 📁 Project Structure

```
├── app.py                    # Main enhanced application
├── search_utils_basic.py     # TF-IDF search engine
├── recommender_basic.py      # Movie recommendation system
├── sentiment_utils.py        # Sentiment analysis utilities
├── gemini_api.py            # Gemini AI integration
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── secrets.toml.example # API key configuration
└── README.md                # This file
```

## 🔧 Configuration

### Streamlit Secrets
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_actual_gemini_api_key_here"
```

### Environment Variables
Alternatively, use `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Deploy on [share.streamlit.io](https://share.streamlit.io)
3. Add `GEMINI_API_KEY` in Streamlit Cloud secrets

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the enhanced app
streamlit run app.py
```

## 📊 Performance Metrics

- ✅ **Search Response**: < 200ms
- ✅ **Graceful Fallback**: Works without torch/transformers
- ✅ **Real-time AI**: Gemini insights and explanations
- ✅ **Intelligent UX**: Autocomplete and fuzzy matching
- ✅ **Portfolio-ready**: Clean, modern interface

## 🎯 Key Features Summary

- **Smart Search**: Autocomplete + fuzzy matching
- **Fast Performance**: Under 200ms response time
- **AI Recommendations**: Content-based with explanations
- **Mood Matching**: Find movies by emotional vibe
- **Gemini AI**: Advanced insights and analysis
- **Graceful Fallback**: Works in any environment
- **Modern UI**: Clean, responsive interface

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the application!

## 📄 License

This project is open source and available under the MIT License.

---

**The enhanced movie sentiment analyzer is now real-time, intelligent, and portfolio-ready!** 🎉
