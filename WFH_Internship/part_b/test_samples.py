import re
import pandas as pd
from sentiment_analyzer import SentimentAnalyzer

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    data = {
        'text': [
            "This product is amazing! Best purchase ever!",
            "Terrible experience, would not recommend to anyone.",
            "It's okay, nothing special about it.",
            "Absolutely love this product, great quality!",
            "Waste of money, very poor quality.",
        ],
        'label': [1, 0, 0, 1, 0]  # 1 for positive, 0 for negative
    }
    return pd.DataFrame(data)

def test_sentiment_analyzer():
    """Test the sentiment analyzer functionality."""
    # Create analyzer instance
    analyzer = SentimentAnalyzer()
    
    # Create and save sample dataset
    df = create_sample_dataset()
    sample_dataset_path = 'sample_dataset.csv'
    df.to_csv(sample_dataset_path, index=False)
    
    # Test preprocessing
    text = "This is a GREAT product! 5/5 would recommend. http://example.com"
    processed = analyzer.preprocess_text(text)
    
    # Verify preprocessing
    assert 'http' not in processed.lower(), "URLs not removed"
    assert processed.islower(), "Text not converted to lowercase"
    assert re.search(r'\w+', processed), "All text removed during preprocessing"
    
    # Test model training and evaluation
    train_df, test_df = analyzer.load_and_prepare_data(sample_dataset_path)
    
    # Verify data preparation
    assert len(train_df) > 0, "Training data is empty"
    assert len(test_df) > 0, "Test data is empty"
    
    # Train the model
    analyzer.train(train_df)
    
    # Evaluate the model
    metrics, report = analyzer.evaluate(test_df)
    
    # Verify metrics
    assert 'accuracy' in metrics, "Accuracy metric missing"
    assert 'precision' in metrics, "Precision metric missing"
    assert 'recall' in metrics, "Recall metric missing"
    
    # Verify metric values
    for metric_name, metric_value in metrics.items():
        assert 0 <= metric_value <= 1, f"{metric_name} out of valid range"
    
    # Test prediction
    sample_text = "This is an excellent product!"
    prediction = analyzer.predict(sample_text)
    assert prediction in [0, 1], f"Invalid prediction: {prediction}"
    
    print("All sentiment analysis test cases passed successfully!")

if __name__ == "__main__":
    test_sentiment_analyzer()
