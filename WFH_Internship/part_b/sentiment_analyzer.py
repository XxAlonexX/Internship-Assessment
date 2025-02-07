import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from typing import Tuple, List

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(random_state=42)
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Join tokens back to string
        return ' '.join(tokens)
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare the dataset."""
        # Load dataset (assuming CSV format with 'text' and 'label' columns)
        df = pd.read_csv(data_path)
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        return train_df, test_df
        
    def train(self, train_df: pd.DataFrame) -> None:
        """Train the sentiment analysis model."""
        # Convert text to TF-IDF features
        X_train = self.vectorizer.fit_transform(train_df['processed_text'])
        y_train = train_df['label']
        
        # Train the model
        self.model.fit(X_train, y_train)
        
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Evaluate the model's performance."""
        # Transform test data
        X_test = self.vectorizer.transform(test_df['processed_text'])
        y_test = test_df['label']
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary')
        }
        
        # Generate detailed classification report
        report = classification_report(y_test, y_pred)
        
        return metrics, report
        
    def predict(self, text: str) -> int:
        """Predict sentiment for new text."""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Transform to TF-IDF features
        X = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction
        
    def predict_batch(self, texts: List[str]) -> List[int]:
        """Predict sentiment for multiple texts."""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform to TF-IDF features
        X = self.vectorizer.transform(processed_texts)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions.tolist()

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Sample dataset path (you'll need to provide your own dataset)
    data_path = "path_to_your_dataset.csv"
    
    # Train the model
    try:
        train_df, test_df = analyzer.load_and_prepare_data(data_path)
        analyzer.train(train_df)
        
        # Evaluate the model
        metrics, report = analyzer.evaluate(test_df)
        
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
            
        print("\nDetailed Classification Report:")
        print(report)
        
        # Example predictions
        sample_texts = [
            "This product is amazing! I love it!",
            "Terrible experience, would not recommend.",
            "It's okay, nothing special."
        ]
        
        print("\nSample Predictions:")
        predictions = analyzer.predict_batch(sample_texts)
        for text, pred in zip(sample_texts, predictions):
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}\n")
            
    except FileNotFoundError:
        print("Please provide a valid path to your dataset.")
