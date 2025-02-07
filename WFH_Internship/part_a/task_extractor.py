import spacy
import nltk
from datetime import datetime
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from typing import Dict, List, Tuple

class TaskExtractor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize task indicators
        self.task_indicators = {
            'verbs': ['need', 'must', 'should', 'has to', 'have to', 'required to'],
            'deadline_markers': ['by', 'before', 'until', 'due'],
            'imperative_starters': ['please', 'kindly', 'ensure', 'make sure']
        }
        
        self.categories = {
            'work': ['report', 'meeting', 'project', 'deadline', 'presentation'],
            'personal': ['clean', 'buy', 'call', 'visit', 'exercise'],
            'urgent': ['immediately', 'asap', 'urgent', 'priority'],
            'routine': ['daily', 'weekly', 'monthly', 'regular']
        }

    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text into sentences."""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        return sentences

    def is_task_sentence(self, sentence: str) -> bool:
        """Determine if a sentence contains a task."""
        doc = self.nlp(sentence.lower())
        
        # Check for task indicators
        for verb in self.task_indicators['verbs']:
            if verb in sentence.lower():
                return True
                
        # Check for imperative mood (sentence starting with a verb)
        if len(doc) > 0 and doc[0].pos_ == 'VERB':
            return True
            
        # Check for deadline markers
        for marker in self.task_indicators['deadline_markers']:
            if marker in sentence.lower():
                return True
                
        return False

    def extract_entity(self, sentence: str) -> str:
        """Extract who needs to perform the task."""
        doc = self.nlp(sentence)
        
        # Look for named entities of type PERSON
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                return ent.text
                
        return None

    def extract_deadline(self, sentence: str) -> str:
        """Extract when the task needs to be completed."""
        doc = self.nlp(sentence)
        
        # Look for time expressions
        for ent in doc.ents:
            if ent.label_ in ['TIME', 'DATE']:
                return ent.text
                
        # Look for deadline markers
        for token in doc:
            if token.text.lower() in self.task_indicators['deadline_markers']:
                # Get the next few tokens as potential deadline
                deadline_phrase = ' '.join([t.text for t in token.rights])
                if deadline_phrase:
                    return deadline_phrase
                    
        return None

    def categorize_task(self, task: str) -> str:
        """Categorize the task based on keywords."""
        task_lower = task.lower()
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in task_lower:
                    return category
                    
        return 'general'

    def extract_tasks(self, text: str) -> List[Dict]:
        """Main function to extract tasks from text."""
        sentences = self.preprocess_text(text)
        tasks = []
        
        for sentence in sentences:
            if self.is_task_sentence(sentence):
                task_info = {
                    'task': sentence,
                    'who': self.extract_entity(sentence),
                    'deadline': self.extract_deadline(sentence),
                    'category': self.categorize_task(sentence)
                }
                tasks.append(task_info)
                
        return tasks

if __name__ == "__main__":
    # Example usage
    text = """Rahul wakes up early every day. He goes to college in the morning and comes back at 3 pm. 
    At present, Rahul is outside. He has to buy the snacks for all of us. 
    Sarah must complete the report by tomorrow morning.
    Please ensure the room is cleaned before 5 PM."""
    
    extractor = TaskExtractor()
    tasks = extractor.extract_tasks(text)
    
    print("\nExtracted Tasks:")
    for task in tasks:
        print("\nTask:", task['task'])
        print("Who:", task['who'] if task['who'] else "Not specified")
        print("Deadline:", task['deadline'] if task['deadline'] else "Not specified")
        print("Category:", task['category'])
