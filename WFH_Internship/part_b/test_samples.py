from sentiment_analyzer import SentimentAnalyzer
import pandas as pd

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
    df.to_csv('sample_dataset.csv', index=False)
    
    # Test preprocessing
    text = "This is a GREAT product! 5/5 would recommend. http://example.com"
    processed = analyzer.preprocess_text(text)
    assert 'http' not in processed.lower()
    assert processed.islower()
    
    # Test model training and evaluation
    train_df, test_df = analyzer.load_and_prepare_data('sample_dataset.csv')
    analyzer.train(train_df)
    metrics, report = analyzer.evaluate(test_df)
    
    # Test prediction
    sample_text = "This is an excellent product!"
    prediction = analyzer.predict(sample_text)
    assert prediction in [0, 1]
    
    print("All test cases passed!")
    print("\nModel Metrics:", metrics)
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    test_sentiment_analyzer()
