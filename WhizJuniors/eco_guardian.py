import openai
import streamlit as st
from dotenv import load_dotenv
import os
import json
import random

# Load environment variables
load_dotenv()

class EcoGuardian:
    def __init__(self):
        self.name = "EcoGuardian"
        self.personality = {
            "role": "environmental AI guardian",
            "tone": "encouraging and informative",
            "mission": "to promote environmental awareness and sustainable living"
        }
        self.knowledge_domains = [
            "climate change",
            "sustainable living",
            "renewable energy",
            "wildlife conservation",
            "waste management"
        ]
        
        # Initialize OpenAI (API key should be in .env file)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_response(self, user_input):
        """Generate AI response using GPT"""
        try:
            system_prompt = f"""
            You are {self.name}, {self.personality['role']}. 
            Your mission is {self.personality['mission']}.
            Respond in a {self.personality['tone']} manner.
            Focus on providing practical, actionable environmental advice.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message['content']
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request. {str(e)}"
    
    def get_eco_tip(self):
        """Generate a random eco-friendly tip"""
        tips = [
            "Try using a reusable water bottle instead of plastic bottles",
            "Switch to LED bulbs to save energy",
            "Start composting your food waste",
            "Use public transportation or bike when possible",
            "Reduce meat consumption to lower your carbon footprint"
        ]
        return random.choice(tips)
    
    def calculate_carbon_impact(self, activity_type, amount):
        """Simple carbon impact calculator"""
        impact_factors = {
            "car_mile": 404,  # grams CO2 per mile
            "plane_mile": 144,  # grams CO2 per mile
            "meat_pound": 6810,  # grams CO2 per pound
            "electricity_kwh": 385  # grams CO2 per kWh
        }
        
        if activity_type in impact_factors:
            return impact_factors[activity_type] * amount
        return 0

    def get_conservation_resources(self):
        """Return list of environmental resources"""
        return [
            "EPA Environmental Education",
            "National Geographic Environment",
            "World Wildlife Fund",
            "Greenpeace",
            "The Nature Conservancy"
        ]
