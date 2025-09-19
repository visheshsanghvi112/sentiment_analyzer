"""
Simple configuration for IMDB sentiment analysis
"""

import os

class Config:
    """Simple configuration class"""
    
    # Model settings
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    MAX_LENGTH = 512
    
    # Dataset settings
    DATASETS = {
        "imdb": {
            "name": "IMDB Movie Reviews",
            "path": "imdb",
            "split": "test",
            "text_col": "text",
            "label_col": "label",
            "label_map": {0: "negative", 1: "positive"}
        },
        "amazon_polarity": {
            "name": "Amazon Product Reviews",
            "path": "amazon_polarity", 
            "split": "test",
            "text_col": "content",
            "label_col": "label",
            "label_map": {0: "negative", 1: "positive"}
        },
        "yelp_polarity": {
            "name": "Yelp Business Reviews",
            "path": "yelp_polarity",
            "split": "test", 
            "text_col": "text",
            "label_col": "label",
            "label_map": {0: "negative", 1: "positive"}
        },
        "rotten_tomatoes": {
            "name": "Rotten Tomatoes Movie Reviews",
            "path": "rotten_tomatoes",
            "split": "test",
            "text_col": "text", 
            "label_col": "label",
            "label_map": {0: "negative", 1: "positive"}
        },
        "sst2": {
            "name": "Stanford Sentiment Treebank",
            "path": "sst2",
            "split": "validation",
            "text_col": "sentence",
            "label_col": "label", 
            "label_map": {0: "negative", 1: "positive"}
        }
    }
    
    CACHE_DIR = ".cache"
    
    # App settings
    APP_TITLE = "Multi-Dataset Sentiment Analyzer"
    
    @classmethod
    def get_cache_dir(cls):
        """Get cache directory"""
        if not os.path.exists(cls.CACHE_DIR):
            os.makedirs(cls.CACHE_DIR)
        return cls.CACHE_DIR