from task_extractor import TaskExtractor

def test_task_extraction():
    """Test the task extraction functionality with various examples."""
    extractor = TaskExtractor()
    
    # Test Case 1: Simple task with person and deadline
    text1 = "Rahul should clean the room by 5 pm today."
    tasks1 = extractor.extract_tasks(text1)
    assert len(tasks1) == 1
    assert tasks1[0]['who'] == 'Rahul'
    assert '5 pm' in tasks1[0]['deadline']
    
    # Test Case 2: Multiple tasks in text
    text2 = """Sarah must complete the report by tomorrow morning.
    Please ensure the room is cleaned before 5 PM.
    John has to submit the project next week."""
    tasks2 = extractor.extract_tasks(text2)
    assert len(tasks2) == 3
    
    # Test Case 3: Text without tasks
    text3 = "The weather is nice today. Birds are chirping in the trees."
    tasks3 = extractor.extract_tasks(text3)
    assert len(tasks3) == 0
    
    # Test Case 4: Task with priority words
    text4 = "Urgently need to send the email to the client."
    tasks4 = extractor.extract_tasks(text4)
    assert len(tasks4) == 1
    assert tasks4[0]['category'] == 'urgent'
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_task_extraction()
