"""
Simple sentiment analysis utilities for IMDB movie reviews
"""

import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from config import Config

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Simple sentiment analyzer using pre-trained models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model"""
        try:
            logger.info(f"Loading model: {Config.MODEL_NAME}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME,
                clean_up_tokenization_spaces=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME)
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def analyze(self, text: str) -> dict:
        """Analyze sentiment of text"""
        if not self.pipeline:
            return {
                'sentiment': 'unknown',
                'confidence': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Clean text
            text = text.strip()
            if not text:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'error': 'Empty text'
                }
            
            # Truncate if too long
            if len(text) > 1000:
                text = text[:1000]
            
            # Get prediction
            results = self.pipeline(text)
            
            # Process results
            if results and len(results) > 0:
                # Find the result with highest score
                best_result = max(results[0], key=lambda x: x['score'])
                
                sentiment = best_result['label'].lower()
                confidence = best_result['score']
                
                # Map labels to standard format
                if sentiment in ['positive', 'pos']:
                    sentiment = 'positive'
                elif sentiment in ['negative', 'neg']:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'text_length': len(text)
                }
            
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': 'No results from model'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def is_available(self) -> bool:
        """Check if analyzer is ready"""
        return self.pipeline is not None