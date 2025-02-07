# NLP Task Extraction and Sentiment Analysis Project

## Overview
This project demonstrates advanced Natural Language Processing (NLP) techniques through two distinct components:

### Part A: Task Extraction System
A sophisticated NLP pipeline designed to identify, extract, and categorize tasks from unstructured text.

#### Key Features
- **Task Identification**: Identifies potential tasks using advanced heuristics
- **Entity Extraction**: Determines who is responsible for the task
- **Deadline Detection**: Extracts temporal information associated with tasks
- **Task Categorization**: Classifies tasks into meaningful categories

#### Techniques Used
- SpaCy for Named Entity Recognition
- Rule-based task identification
- Custom tokenization and preprocessing
- Dynamic task categorization

### Part B: Sentiment Analysis Model
A machine learning model for classifying customer reviews as positive or negative.

#### Key Features
- **Text Preprocessing**: Comprehensive text cleaning and normalization
- **Feature Extraction**: TF-IDF vectorization
- **Classification**: Logistic Regression model
- **Performance Metrics**: Accuracy, precision, and recall evaluation

#### Techniques Used
- TF-IDF feature extraction
- Logistic Regression classification
- Custom stop words handling
- Train-test split for model validation

## Project Structure
```
├── part_a/
│   ├── task_extractor.py       # Core task extraction logic
│   ├── utils.py                # Utility functions for Part A
│   └── test_samples.py         # Test cases for task extraction
│
├── part_b/
│   ├── sentiment_analyzer.py   # Sentiment classification model
│   ├── utils.py                # Utility functions for Part B
│   └── test_samples.py         # Test cases for sentiment analysis
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Task Extraction
```python
from part_a.task_extractor import TaskExtractor

extractor = TaskExtractor()
text = "Rahul should clean the room by 5 pm today."
tasks = extractor.extract_tasks(text)
```

### Sentiment Analysis
```python
from part_b.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
review = "This product is amazing!"
sentiment = analyzer.predict(review)
```

## Running Tests
```bash
# Run Part A tests
python part_a/test_samples.py

# Run Part B tests
python part_b/test_samples.py
```

## Dependencies
- SpaCy
- scikit-learn
- pandas
- numpy

## Limitations and Future Improvements
- Expand task extraction heuristics
- Improve entity recognition accuracy
- Add support for more complex sentiment analysis models
- Implement more comprehensive preprocessing techniques

## License
[Specify your license here]

## Contact
[Your contact information]
