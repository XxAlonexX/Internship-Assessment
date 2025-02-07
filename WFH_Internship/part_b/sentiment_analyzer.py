import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from typing import Tuple, List, Dict

class SentimentAnalyzer:
    def __init__(self):
        # Stop words list (manually defined)
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
            'with', 'about', 'against', 'between', 'into', 'through', 'during', 
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
            'should', 'now'
        }
        
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(random_state=42)
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize (split into words)
        tokens = text.split()
        
        # Remove stop words
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
        
    def evaluate(self, test_df: pd.DataFrame) -> Tuple[Dict[str, float], str]:
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
    data_path = "sample_dataset.csv"
    
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
        print("Please create a sample_dataset.csv with 'text' and 'label' columns.")
