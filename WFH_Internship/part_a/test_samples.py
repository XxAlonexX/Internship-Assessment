import re
from task_extractor import TaskExtractor

def manual_sentence_tokenize(text):
    """Manually split text into sentences."""
    # Split on common sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def test_task_extraction():
    """Test the task extraction functionality with various examples."""
    extractor = TaskExtractor()
    
    # Test Case 1: Simple task with person and deadline
    text1 = "Rahul should clean the room by 5 pm today."
    tasks1 = extractor.extract_tasks(text1)
    assert len(tasks1) == 1, f"Expected 1 task, got {len(tasks1)}"
    
    # Flexible entity extraction
    assert tasks1[0]['who'] is not None, "Failed to extract entity"
    assert re.search(r'Rahul', tasks1[0]['who'], re.IGNORECASE), f"Expected Rahul, got {tasks1[0]['who']}"
    
    # Flexible deadline extraction
    assert tasks1[0]['deadline'] is not None, "Failed to extract deadline"
    assert re.search(r'5 pm', tasks1[0]['deadline'], re.IGNORECASE), f"Expected '5 pm', got {tasks1[0]['deadline']}"
    
    # Test Case 2: Multiple tasks in text
    text2 = """Sarah must complete the report by tomorrow morning.
    Please ensure the room is cleaned before 5 PM.
    John has to submit the project next week."""
    tasks2 = extractor.extract_tasks(text2)
    assert len(tasks2) >= 3, f"Expected at least 3 tasks, got {len(tasks2)}"
    
    # Test Case 3: Text without tasks
    text3 = "The weather is nice today. Birds are chirping in the trees."
    tasks3 = extractor.extract_tasks(text3)
    assert len(tasks3) == 0, f"Expected 0 tasks, got {len(tasks3)}"
    
    # Test Case 4: Task with priority words
    text4 = "Urgently need to send the email to the client."
    tasks4 = extractor.extract_tasks(text4)
    assert len(tasks4) == 1, f"Expected 1 task, got {len(tasks4)}"
    assert tasks4[0]['category'] == 'urgent', f"Expected 'urgent' category, got {tasks4[0]['category']}"
    
    print("All test cases passed successfully!")

if __name__ == "__main__":
    test_task_extraction()
