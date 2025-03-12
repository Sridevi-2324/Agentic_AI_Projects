import os
import openai
import requests
import langgraph
import langsmith
import faiss
import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langchain.schema import AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b", temperature=0.7)

# Define State Schema with region added
class HealthQueryState(BaseModel):
    gender: str
    age: int
    weight: float
    target_weight: float
    height: float
    lifestyle: str
    meal_preferences: str
    occupation: str
    region: str  # New field for region
    fitness_goals: str
    personalized_plan: str = ""
    user_feedback: str = ""

# Define Workflow Functions
def analyze_query(state: HealthQueryState):
    return state

def retrieve_information(state: HealthQueryState):
    prompt = f"""
    You are a professional dietitian and fitness expert. Generate a personalized health and diet plan for a {state.age}-year-old {state.gender} from {state.region} 
    who weighs {state.weight}kg, is {state.height}cm tall, follows a {state.lifestyle} lifestyle, and works as a {state.occupation}. 
    
    The user prefers {state.meal_preferences} meals and aims for {state.fitness_goals}. The target weight is {state.target_weight}kg. 
    
    Adjust the diet and workout plan based on their age, occupation, regional dietary habits, and fitness goal. 
    Provide a detailed daily/weekly schedule with estimated timeframes for achieving the target weight & also include meal timings.
    """
    llm_response = llm.invoke(prompt)
    state.personalized_plan = llm_response.content if isinstance(llm_response, AIMessage) else str(llm_response)
    return state

def human_feedback(state: HealthQueryState):
    if state.user_feedback:
        prompt = f"Modify the following health plan based on user feedback:\n\nUser Feedback: {state.user_feedback}\n\nOriginal Plan:\n{state.personalized_plan}"
        llm_response = llm.invoke(prompt)
        state.personalized_plan = llm_response.content if isinstance(llm_response, AIMessage) else str(llm_response)
    return state

def create_healthcare_rag_workflow():
    graph = StateGraph(HealthQueryState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve_information", retrieve_information)
    graph.add_node("human_feedback", human_feedback)
    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve_information")
    graph.add_edge("retrieve_information", "human_feedback")
    return graph.compile()

# Create Workflow
app = create_healthcare_rag_workflow()

# Streamlit UI
st.title("💪🏋️‍♂️🏃‍♀️ AI Dietician Assistant  🚴‍♂️🤸‍♂️🧘‍♀️")
st.subheader("Generate a Personalized Diet & Activity Plan")

# Session state
st.session_state.setdefault("response_state", None)
st.session_state.setdefault("user_feedback", "")

# User Inputs
gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])
age = st.number_input("Enter your age", 10, 100, step=1)
weight = st.number_input("Enter your weight (kg)", 30.0, 200.0, step=0.1)
target_weight = st.number_input("Enter your target weight (kg)", 30.0, 200.0, step=0.1)
height = st.number_input("Enter your height (cm)", 100.0, 250.0, step=0.1)
lifestyle = st.selectbox("Select Your Lifestyle", [
  "Active Lifestyle (Prioritizes fitness, regular exercise, and an active daily routine)",
  "Healthy Eating Lifestyle (Focuses on balanced nutrition, whole foods, and mindful eating)",
  "Holistic Lifestyle (Incorporates natural living, meditation, and alternative health practices)",
  "Corporate/Professional Lifestyle (Structured around career growth, business success, and work-life balance)",
  "Freelance/Gig Lifestyle (Emphasizes flexible work, self-employment, and project-based jobs)",
  "Minimalist Lifestyle (Prioritizes simplicity, reducing material possessions, and focusing on experiences)",
  "Family-Oriented Lifestyle (Centers around parenting, family bonding, and home life)",
  "Social/Extroverted Lifestyle (Focuses on social interactions, networking, and group activities)",
  "Introverted Lifestyle (Prefers solitude, quiet activities, and deep personal connections)",
  "Nomadic Lifestyle (Constantly traveling, moving from place to place, often working remotely)",
  "Backpacker/Explorer Lifestyle (Prioritizes adventure, new experiences, and cultural immersion)",
  "Sustainable/Green Living (Focuses on reducing environmental impact and eco-friendly habits)",
  "Off-Grid Lifestyle (Living independently, using renewable energy, and self-sufficiency)",
  "Digital Nomad (Works remotely while traveling, relying on technology for income)",
  "Tech-Driven Lifestyle (Heavily integrates smart devices, AI, and automation into daily life)",
  "Artist/Creative Lifestyle (Centers around self-expression, art, music, or writing)",
  "DIY/Handmade Lifestyle (Focuses on crafting, making things by hand, and self-sufficiency)"
])
meal_preferences = st.selectbox("Select Your Meal Preference", ["Vegetarian", "Non-Vegetarian", "Vegan", "Keto", "Mediterranean"])
occupation = st.selectbox("Select Your Occupation", ["Techie (works 8 hrs)", "Farmer", "Homemaker", "Student", "Athlete", "Other"])
region = st.selectbox("Select Your Region", ["Asian", "South Indian", "North India", "African", "European", "North American", "South American", "Australian", "Middle Eastern"])
fitness_goals = st.selectbox("Select Your Fitness Goal", ["Weight Loss", "Muscle Gain", "General Fitness", "Endurance Training"])

if st.button("Generate Plan"):
    user_state = HealthQueryState(
        gender=gender, age=age, weight=weight, target_weight=target_weight, height=height,
        lifestyle=lifestyle, meal_preferences=meal_preferences, occupation=occupation, region=region, fitness_goals=fitness_goals
    )
    st.session_state.response_state = HealthQueryState(**dict(app.invoke(user_state)))

if st.session_state.response_state:
    st.subheader("📋 Your Personalized Health Plan")
    st.write(st.session_state.response_state.personalized_plan)
    
    # User Feedback
    st.session_state.user_feedback = st.text_area("Customize your diet plan:", st.session_state.user_feedback, key="feedback_input")
    if st.button("Customize Plan"):
        st.session_state.response_state.user_feedback = st.session_state.user_feedback
        st.session_state.response_state = HealthQueryState(**dict(app.invoke(st.session_state.response_state)))
        st.rerun()