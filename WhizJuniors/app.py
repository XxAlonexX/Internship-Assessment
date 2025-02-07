import streamlit as st
from eco_guardian import EcoGuardian
import plotly.graph_objects as go

# Initialize EcoGuardian
guardian = EcoGuardian()

# Set page config
st.set_page_config(
    page_title="EcoGuardian AI",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8f4;
    }
    .main-header {
        color: #2e7d32;
        text-align: center;
    }
    .chat-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>🌍 EcoGuardian AI</h1>", unsafe_allow_html=True)
st.markdown("### Your Personal Environmental Awareness Assistant")

# Sidebar
with st.sidebar:
    st.header("🌱 Daily Eco Tip")
    st.info(guardian.get_eco_tip())
    
    st.header("📚 Resources")
    for resource in guardian.get_conservation_resources():
        st.write(f"- {resource}")

# Main chat interface
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask EcoGuardian about environmental topics..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = guardian.generate_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("</div>", unsafe_allow_html=True)

# Carbon Impact Calculator
st.header("🌡️ Carbon Impact Calculator")
col1, col2 = st.columns(2)

with col1:
    activity = st.selectbox(
        "Select activity",
        ["car_mile", "plane_mile", "meat_pound", "electricity_kwh"]
    )
    amount = st.number_input("Enter amount", min_value=0.0, value=1.0)

with col2:
    if st.button("Calculate Impact"):
        impact = guardian.calculate_carbon_impact(activity, amount)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = impact,
            title = {'text': "Carbon Impact (g CO2)"},
            gauge = {'axis': {'range': [None, impact*2]},
                    'bar': {'color': "#2e7d32"}}
        ))
        st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Built with 💚 by EcoGuardian AI | Helping create a sustainable future")
