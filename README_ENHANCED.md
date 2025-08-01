# 🎬 Enhanced AI Movie Sentiment Analyzer

A powerful, intelligent movie sentiment analyzer with real-time search, AI recommendations, and advanced analytics powered by Google Gemini.

## 🚀 Features

### 🔍 Smart Search & Autocomplete
- **Intelligent Autocomplete**: Type movie names and get instant suggestions
- **Fuzzy Matching**: Handles typos like "tightanic" → "Titanic"
- **Fast Performance**: Search results in under 200ms
- **Real-time Suggestions**: Dynamic dropdown with matching movies

### 🎯 AI-Powered Recommendations
- **Content-Based Filtering**: Uses BERT embeddings for similarity
- **Movie-to-Movie Recommendations**: Find similar movies based on review content
- **Review-Based Recommendations**: Get movies based on your review text
- **Mood-Based Search**: Find movies matching your desired mood/feeling
- **AI Explanations**: Gemini explains why movies are recommended

### 📊 Advanced Analytics
- **Sentiment Analysis**: BERT + VADER for comprehensive analysis
- **Emotion Detection**: Identify joy, anger, sadness, fear, surprise, disgust
- **Review Analysis**: Top positive/negative reviews with AI insights
- **Word Clouds**: Visual representation of common themes
- **Sentiment Distribution**: Charts showing review sentiment patterns

### 🤖 Gemini AI Integration
- **Emotional Insights**: Deep analysis of movie emotional impact
- **Recommendation Explanations**: Why movies are similar
- **Mood Analysis**: AI analysis of mood-based recommendations
- **Review Pattern Analysis**: Identify trends in movie reviews
- **Custom Questions**: Ask Gemini anything about movies and reviews

### 💭 Mood-Based Search
- **Emotional Matching**: Find movies for specific moods
- **Vibe Descriptions**: "feel-good", "dark", "romantic", "inspiring"
- **AI Analysis**: Gemini explains mood-movie connections
- **Personalized Results**: Based on emotional content analysis

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sentiment_analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Or set up Streamlit secrets in `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_actual_gemini_api_key_here"
```

### 4. Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your environment variables

## 🚀 Usage

### Run the Enhanced App
```bash
streamlit run app_enhanced.py
```

### Test the System
```bash
python test_enhanced.py
```

## 📱 App Pages

### 🔍 Smart Movie Search
- **Autocomplete Mode**: Type and get instant suggestions
- **Fuzzy Search Mode**: Find movies even with typos
- **Movie Analysis**: Comprehensive stats and reviews
- **AI Insights**: Emotional analysis with Gemini

### 📊 Review Analysis
- **Custom Reviews**: Analyze your own movie reviews
- **Multiple Methods**: BERT + VADER sentiment analysis
- **Emotion Detection**: Identify emotional content
- **AI Explanations**: Detailed analysis with Gemini

### 🎯 Movie Recommendations
- **Movie-Based**: Find similar movies to ones you like
- **Review-Based**: Get recommendations from your review text
- **Movie Clusters**: Discover movies grouped by themes
- **AI Explanations**: Why movies are recommended

### 💭 Mood-Based Search
- **Mood Input**: Describe your desired feeling
- **Emotional Matching**: Find movies for your mood
- **AI Analysis**: Gemini explains mood-movie connections
- **Personalized Results**: Based on emotional content

### 🤖 AI Insights
- **Quick Questions**: Pre-defined analysis questions
- **Custom Questions**: Ask Gemini anything about movies
- **Pattern Analysis**: Identify review trends
- **Industry Insights**: Movie industry analysis

### 📈 Dataset Explorer
- **Browse Reviews**: Explore the IMDB dataset
- **Filter by Movie**: Find reviews for specific movies
- **Visualizations**: Sentiment charts and word clouds
- **Review Display**: Read actual movie reviews

## 🏗️ Architecture

### Core Modules

#### `search_utils.py`
- **MovieSearchEngine**: Fast autocomplete and fuzzy search
- **EmbeddingEngine**: BERT embeddings for similarity
- **TextProcessor**: Language detection and text cleaning

#### `recommender.py`
- **MovieRecommender**: Content-based recommendation system
- **Similarity Calculation**: Cosine similarity with embeddings
- **Feature Extraction**: Movie characteristics analysis

#### `gemini_utils.py`
- **AI Explanations**: Detailed recommendation explanations
- **Mood Analysis**: Emotional content analysis
- **Review Insights**: Deep review pattern analysis

#### `app_enhanced.py`
- **Main Application**: Streamlit interface
- **Page Management**: Multi-page navigation
- **Real-time Features**: Fast, responsive UI

## 🎯 Key Features Explained

### Autocomplete + Fuzzy Matching
```python
# Fast autocomplete with fuzzy matching
suggestions = search_engine.autocomplete("titanic", limit=10)
fuzzy_matches = search_engine.fuzzy_search("tightanic", threshold=70)
```

### AI Recommendations
```python
# Get similar movies with AI explanations
recommendations = recommender.find_similar_movies("Titanic", top_k=5)
explanation = explain_movie_recommendation(source, target, features, similarity)
```

### Mood-Based Search
```python
# Find movies for specific moods
recommendations = recommender.get_mood_based_recommendations("feel-good", top_k=5)
analysis = analyze_mood_based_search("feel-good", recommendations)
```

### Emotional Analysis
```python
# Get emotional insights for movies
insights = get_emotional_insights_for_movie("Titanic", reviews_df)
```

## 🔧 Performance Optimizations

### Fast Search (< 200ms)
- **RapidFuzz**: Optimized fuzzy string matching
- **Caching**: Embedding and search result caching
- **Efficient Algorithms**: Optimized similarity calculations

### Real-time Features
- **Streamlit Caching**: Smart data caching
- **Lazy Loading**: Load models only when needed
- **Background Processing**: Non-blocking AI operations

### Memory Management
- **Embedding Cache**: Reuse computed embeddings
- **Smart Loading**: Load only necessary data
- **Garbage Collection**: Automatic memory cleanup

## 🎨 UI/UX Features

### Modern Design
- **Gradient Headers**: Beautiful visual design
- **Card Layouts**: Clean, organized information display
- **Color-coded Sentiment**: Visual sentiment indicators
- **Responsive Layout**: Works on all screen sizes

### Interactive Elements
- **Expandable Reviews**: Click to read full reviews
- **Dynamic Charts**: Interactive visualizations
- **Real-time Updates**: Instant search results
- **Progress Indicators**: Loading states for long operations

### User Experience
- **Intuitive Navigation**: Clear page structure
- **Helpful Tooltips**: Context-sensitive help
- **Error Handling**: Graceful error messages
- **Performance Feedback**: Loading times and status

## 🔍 Search Capabilities

### Autocomplete Examples
- `"titanic"` → ["Titanic", "Titanic (1997)", ...]
- `"godfather"` → ["The Godfather", "Godfather", ...]
- `"star wars"` → ["Star Wars", "Star Wars Episode IV", ...]

### Fuzzy Matching Examples
- `"tightanic"` → "Titanic" (similarity: 85%)
- `"godfatha"` → "The Godfather" (similarity: 78%)
- `"starwars"` → "Star Wars" (similarity: 92%)

## 🎯 Recommendation Examples

### Movie-Based Recommendations
**Input**: "Titanic"
**Output**: 
1. "The Notebook" (similarity: 0.82) - Romantic dramas with emotional depth
2. "Romeo + Juliet" (similarity: 0.79) - Tragic love stories
3. "Gone with the Wind" (similarity: 0.76) - Epic romantic films

### Mood-Based Recommendations
**Input**: "feel-good"
**Output**:
1. "The Princess Bride" (match: 0.85) - Uplifting adventure
2. "Forrest Gump" (match: 0.83) - Heartwarming story
3. "The Sound of Music" (match: 0.81) - Joyful musical

## 🤖 AI Analysis Examples

### Emotional Insights
```
"Titanic" creates an emotional journey that takes viewers from 
excitement and romance to heartbreak and tragedy. The film 
resonates with audiences through its universal themes of love, 
loss, and sacrifice, making it emotionally powerful despite 
its tragic ending.
```

### Recommendation Explanations
```
"The Notebook" is recommended for "Titanic" fans because both 
movies share themes of passionate, doomed romance and emotional 
intensity. They both feature strong emotional storytelling, 
beautiful cinematography, and memorable love stories that 
resonate deeply with audiences.
```

## 📊 Analytics Features

### Sentiment Analysis
- **BERT Model**: State-of-the-art transformer model
- **VADER**: Traditional sentiment analysis
- **Confidence Scores**: Reliability indicators
- **Multi-language Support**: Language detection and translation

### Emotion Detection
- **6 Emotions**: Joy, Anger, Sadness, Fear, Surprise, Disgust
- **Keyword Analysis**: Emotion-specific word detection
- **Intensity Scoring**: Emotional intensity measurement

### Review Analysis
- **Top Reviews**: Best positive and negative reviews
- **Keyword Extraction**: Important themes and topics
- **Length Analysis**: Review length patterns
- **Language Detection**: Multi-language support

## 🔧 Technical Details

### Dependencies
```
streamlit>=1.28.0
transformers>=4.30.0
torch>=1.13.0
rapidfuzz>=3.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
google-generativeai>=0.3.0
```

### Performance Metrics
- **Search Speed**: < 200ms for autocomplete
- **Recommendation Time**: < 2s for 5 recommendations
- **AI Analysis**: < 5s for detailed insights
- **Memory Usage**: Optimized for large datasets

### Scalability
- **Caching**: Smart result caching
- **Lazy Loading**: Load models on demand
- **Background Processing**: Non-blocking operations
- **Memory Management**: Efficient resource usage

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API key

# Run the app
streamlit run app_enhanced.py
```

### Streamlit Cloud Deployment
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard
4. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_enhanced.py"]
```

## 🧪 Testing

### Run Tests
```bash
python test_enhanced.py
```

### Test Coverage
- ✅ Module imports
- ✅ Data loading
- ✅ Search functionality
- ✅ Recommendation system
- ✅ Text processing
- ✅ Embedding generation
- ✅ Sentiment analysis

## 📈 Future Enhancements

### Planned Features
- **User Profiles**: Personalized recommendations
- **Watch History**: Track viewed movies
- **Social Features**: Share recommendations
- **Advanced Analytics**: More detailed insights
- **Mobile App**: Native mobile application

### Technical Improvements
- **Database Integration**: Persistent user data
- **Real-time Updates**: Live recommendation updates
- **Advanced ML**: More sophisticated algorithms
- **API Endpoints**: RESTful API for external use

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write comprehensive tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the IMDB dataset
- **Google Gemini**: For AI insights and analysis
- **Streamlit**: For the web framework
- **Sentence Transformers**: For embedding models
- **RapidFuzz**: For fast fuzzy matching

---

**🎬 Ready to discover amazing movies with AI-powered insights!** 