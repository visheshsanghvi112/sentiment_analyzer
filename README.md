# 🎬 IMDB Sentiment Analysis App

A comprehensive Streamlit web application for advanced sentiment analysis using movie reviews from the IMDB dataset.

## 🚀 Features

- **Dataset Explorer**: Browse and filter movie reviews from the IMDB dataset
- **Custom Review Analysis**: Analyze your own movie reviews with multiple methods
- **Advanced NLP**: Uses both BERT-based models and VADER sentiment analysis
- **Emotion Detection**: Identifies emotions like joy, anger, sadness, etc.
- **AI Insights**: Powered by Google's Gemini AI for deeper analysis
- **Visualizations**: Word clouds and sentiment distribution charts

## 📦 Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gemini API** (optional but recommended):
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Copy `.env.example` to `.env`
   - Add your API key to the `.env` file

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage

### 📊 Dataset Explorer
- View random samples from the IMDB dataset
- Filter reviews by movie name
- See sentiment distribution charts
- Generate word clouds from reviews

### 🧠 Analyze Review
- Enter your own movie review
- Get sentiment analysis with confidence scores
- See VADER sentiment scores
- Detect emotions in the text

### 🤖 Gemini Insights
- Ask AI questions about movie reviews
- Get explanations for sentiment predictions
- Discover patterns in movie reviews

## 🛠️ Technical Details

### Models Used
- **BERT**: `distilbert-base-uncased-finetuned-sst-2-english`
- **VADER**: Traditional rule-based sentiment analysis
- **Emotion Detection**: Keyword-based emotion classification

### Dataset
- **IMDB Movie Reviews**: 50,000 movie reviews from Hugging Face datasets
- **Labels**: Binary sentiment (positive/negative)

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── sentiment_utils.py     # Sentiment analysis functions
├── gemini_api.py         # Gemini AI integration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Streamlit Secrets
For deployment, you can also use Streamlit secrets:
```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your `GEMINI_API_KEY` in the secrets section

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the application!

## 📄 License

This project is open source and available under the MIT License.
