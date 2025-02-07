import re
import numpy as np
from typing import List, Dict
from collections import Counter

def clean_text(text: str) -> str:
    """Remove special characters and normalize text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_word_stats(texts: List[str]) -> Dict:
    """Calculate basic word statistics from a list of texts."""
    # Combine all texts
    all_words = ' '.join(texts).split()
    
    # Calculate statistics
    stats = {
        'total_words': len(all_words),
        'unique_words': len(set(all_words)),
        'avg_words_per_text': len(all_words) / len(texts),
        'most_common_words': Counter(all_words).most_common(10)
    }
    
    return stats

def calculate_class_weights(labels: List[int]) -> Dict[int, float]:
    """Calculate balanced class weights for imbalanced datasets."""
    counts = Counter(labels)
    total = len(labels)
    
    class_weights = {}
    for label, count in counts.items():
        class_weights[label] = total / (len(counts) * count)
        
    return class_weights

def get_top_features(vectorizer, model, n_top=10) -> Dict[str, List[str]]:
    """Get top features (words) contributing to positive and negative sentiment."""
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
    else:
        return None
        
    # Get indices of top positive and negative coefficients
    top_positive_idx = np.argsort(coefficients)[-n_top:]
    top_negative_idx = np.argsort(coefficients)[:n_top]
    
    # Get the corresponding feature names
    top_features = {
        'positive': [feature_names[i] for i in top_positive_idx],
        'negative': [feature_names[i] for i in top_negative_idx]
    }
    
    return top_features
