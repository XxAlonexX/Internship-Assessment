import re
from typing import List
from datetime import datetime, timedelta

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def parse_relative_date(text: str) -> str:
    """Convert relative date expressions to actual dates."""
    text = text.lower()
    today = datetime.now()
    
    # Handle common relative date expressions
    if 'tomorrow' in text:
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    elif 'next week' in text:
        return (today + timedelta(weeks=1)).strftime('%Y-%m-%d')
    elif 'today' in text:
        return today.strftime('%Y-%m-%d')
        
    return text

def extract_time(text: str) -> str:
    """Extract time expressions from text."""
    # Match patterns like "5 pm", "3:00", "15:00", etc.
    time_pattern = r'\b(?:(?:0?[1-9]|1[0-2])(?::(?:[0-5][0-9]))?\s*[ap]\.?m\.?|(?:[01]?[0-9]|2[0-3]):[0-5][0-9])\b'
    match = re.search(time_pattern, text, re.IGNORECASE)
    return match.group() if match else None

def get_task_priority(text: str) -> str:
    """Determine task priority based on keywords."""
    priority_keywords = {
        'high': ['urgent', 'asap', 'immediately', 'critical'],
        'medium': ['soon', 'important'],
        'low': ['when possible', 'eventually']
    }
    
    text_lower = text.lower()
    for priority, keywords in priority_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return priority
            
    return 'normal'
