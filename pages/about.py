import streamlit as st

# About/Help Page
def about():
    st.title("About OratorAI")
    st.write("**OratorAI** is a tool that analyzes videos to extract meaningful insights.")
    st.subheader("How it works:")
    st.write("1. **Emotion Analysis**: Detects emotions through facial expressions.")
    st.write("2. **Transcript Analysis**: Analyzes text for various traits.")
    st.write("3. **Speech Analysis**: Evaluates speech rate and filler words.")
    st.write("For more information, contact: support@oratorai.com")
