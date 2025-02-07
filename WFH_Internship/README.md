# NLP Task Extraction and Sentiment Analysis Project

This project consists of two main parts:

## Part A: Task Extraction System (70 marks)
An NLP-based system that extracts and categorizes tasks from unstructured text using heuristic approaches.

### Features:
- Task identification from unstructured text
- Entity extraction (who needs to do the task)
- Deadline extraction (when the task needs to be completed)
- Task categorization using word embeddings

## Part B: Sentiment Analysis Model (30 marks)
A machine learning model for classifying customer reviews as positive or negative.

### Features:
- Text preprocessing pipeline
- TF-IDF feature extraction
- Machine learning classification model
- Model evaluation metrics

## Setup and Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── part_a/
│   ├── task_extractor.py
│   ├── utils.py
│   └── test_samples.py
├── part_b/
│   ├── sentiment_analyzer.py
│   ├── utils.py
│   └── model_training.py
├── requirements.txt
└── README.md
```

## Usage

### Part A: Task Extraction
```python
from part_a.task_extractor import TaskExtractor

extractor = TaskExtractor()
text = "Rahul should clean the room by 5 pm today."
tasks = extractor.extract_tasks(text)
```

### Part B: Sentiment Analysis
```python
from part_b.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
review = "This product is amazing!"
sentiment = analyzer.predict(review)
```
