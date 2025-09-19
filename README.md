# 🎬 IMDB Movie Sentiment Analyzer

A simple, clean Streamlit web application for analyzing movie review sentiment using machine learning.

## 📁 Project Structure

```
sentiment_analyzer/
├── 📱 Core Application
│   ├── app.py                    # Main Streamlit application
│   ├── run_app.py               # Application launcher
│   └── config.py                # Simple configuration
│
├── 🧠 ML Components
│   ├── sentiment_utils.py       # Sentiment analysis logic
│   └── utils.py                 # Helper utilities
│
├── 🔧 Setup
│   ├── logger_config.py         # Logging setup
│   ├── requirements.txt         # Dependencies
│   └── README.md               # This file
│
└── 📂 Data
    └── logs/                   # Application logs
```

## 🚀 Features

- **Simple Interface**: Clean, easy-to-use Streamlit interface
- **IMDB Dataset**: Uses real IMDB movie review data (up to 500 reviews)
- **Pre-trained Models**: Leverages DistilBERT for sentiment analysis
- **Real-time Analysis**: Analyze custom text or sample data
- **Accuracy Metrics**: Test on 250+ reviews for reliable results
- **Configurable Sample Size**: Choose 50-500 reviews for testing

## 🛠️ Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**:
   ```bash
   python run_app.py
   ```
4. **Open browser**: Navigate to `http://localhost:8501`

## 📖 Usage

### Analyze Custom Text
1. Go to the "📝 Analyze Text" tab
2. Enter or paste a movie review
3. Click "🔍 Analyze" to get sentiment prediction

### Test with Sample Data
1. Go to the "📊 Sample Data" tab  
2. Click "🔍 Analyze Sample Reviews"
3. See accuracy metrics and detailed results

## 🔧 Technical Details

- **Model**: DistilBERT (fine-tuned for sentiment analysis)
- **Dataset**: IMDB movie reviews
- **Framework**: Streamlit + Transformers + PyTorch
- **Sentiments**: Positive, Negative, Neutral

## 📊 What You Get

- **Sentiment Prediction**: Positive/Negative classification
- **Confidence Score**: How certain the model is
- **Accuracy Metrics**: Performance on sample data
- **Clean Interface**: No clutter, just results

## 🚨 Requirements

- Python 3.8+
- Internet connection (for downloading models)
- ~2GB disk space (for model cache)

## 🔧 Troubleshooting

### JavaScript/Browser Issues
If you see `TypeError: Failed to fetch dynamically imported module` errors:

1. **Clear browser cache** (Ctrl+Shift+Delete)
2. **Try incognito/private mode** (Ctrl+Shift+N)
3. **Try a different browser** (Chrome, Firefox, Edge)
4. **Restart browser completely**
5. **Try different launch methods:**
   ```bash
   # Standard launcher
   python run_app.py
   
   # Direct launcher
   python run_direct.py
   
   # Manual Streamlit
   streamlit run app.py
   
   # Different port
   streamlit run app.py --server.port 8502
   ```

### Test Browser Compatibility
Open `test_browser.html` in your browser to check JavaScript compatibility.

## 🎯 Perfect For

- Learning sentiment analysis
- Testing movie review classification
- Understanding ML model predictions
- Simple sentiment analysis tasks

---

**That's it!** A clean, simple sentiment analyzer focused on doing one thing well. 🎬