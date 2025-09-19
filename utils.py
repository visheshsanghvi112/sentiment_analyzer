"""
Simple utilities for the sentiment analyzer
"""

import re
import logging
from datasets import load_dataset
from config import Config

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean text for analysis"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip and return
    return text.strip()

def load_dataset_sample(dataset_key: str = "imdb", num_samples: int = 250) -> list:
    """Load sample data from specified dataset"""
    try:
        if dataset_key not in Config.DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            return []
        
        dataset_config = Config.DATASETS[dataset_key]
        dataset_name = dataset_config["name"]
        
        logger.info(f"Loading up to {num_samples} samples from {dataset_name}...")
        
        # Load dataset
        dataset = load_dataset(
            dataset_config["path"], 
            split=dataset_config["split"], 
            streaming=True
        )
        
        samples = []
        text_col = dataset_config["text_col"]
        label_col = dataset_config["label_col"]
        label_map = dataset_config["label_map"]
        
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            
            # Get text and label
            text = example.get(text_col, "")
            label_idx = example.get(label_col, 0)
            
            # Clean and validate text
            cleaned_text = clean_text(str(text))
            if cleaned_text and len(cleaned_text) > 10:  # Only add meaningful reviews
                samples.append({
                    'text': cleaned_text,
                    'label': label_map.get(label_idx, 'unknown'),
                    'dataset': dataset_name
                })
        
        actual_count = len(samples)
        if actual_count < num_samples:
            logger.warning(f"Only {actual_count} samples available from {dataset_name} (requested {num_samples})")
        else:
            logger.info(f"Successfully loaded {actual_count} samples from {dataset_name}")
        
        return samples
        
    except Exception as e:
        logger.error(f"Error loading {dataset_key} data: {e}")
        return []

def load_mixed_samples(num_samples_per_dataset: int = 50) -> list:
    """Load samples from all available datasets"""
    all_samples = []
    
    for dataset_key in Config.DATASETS.keys():
        samples = load_dataset_sample(dataset_key, num_samples_per_dataset)
        all_samples.extend(samples)
        
    logger.info(f"Loaded total of {len(all_samples)} samples from {len(Config.DATASETS)} datasets")
    return all_samples

# Keep backward compatibility
def load_imdb_sample(num_samples: int = 250) -> list:
    """Load IMDB samples (backward compatibility)"""
    return load_dataset_sample("imdb", num_samples)

def format_confidence(confidence: float) -> str:
    """Format confidence as percentage"""
    return f"{confidence * 100:.1f}%"

def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment display"""
    colors = {
        'positive': '#28a745',  # Green
        'negative': '#dc3545',  # Red
        'neutral': '#6c757d'    # Gray
    }
    return colors.get(sentiment, '#6c757d')