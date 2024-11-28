import streamlit as st

# Download Page
def download():
    st.title("Download Analysis Report")
    st.write("You can download the full report or specific sections.")
    
    if st.button("Download Full Report"):
        st.success("Download started!")
    st.download_button("Download Emotion Analysis", data=b"Emotion data", file_name="emotion_report.csv")
    st.download_button("Download Transcript Analysis", data=b"Transcript data", file_name="transcript_report.csv")
    st.download_button("Download Speech Analysis", data=b"Speech data", file_name="speech_report.csv")
