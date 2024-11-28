import streamlit as st
import os
from helper import extract_snapshots_for_sentiment, extract_audio_and_duration, predict_emotion_from_image
import matplotlib.pyplot as plt



def results():
    st.title("Analysis Results")
    
    temp_folder = os.path.join(os.path.dirname(__file__), 'audio.wav')  # Path to 'temp' folder

    audio_path = os.path.join(os.path.dirname(__file__), 'temp/recorded_audio.wav')
    video_path = os.path.join(os.path.dirname(__file__), 'temp/recorded_video.mp4')
    
    # Inform the user if files are not present
    if not os.path.exists(audio_path) or not os.path.exists(video_path):
        st.error("Required files (audio and video) are missing. Please make sure the recording was done correctly.")
        return

    # Add a button to trigger analysis
    if st.button("Run Analysis"):
        try:
            # transcript = extract_transcript(audio_path) - Will be implemented in next push
            # analysis_results = extract_snapshots_for_sentiment(video_path)
            audio_analysis = extract_audio_and_duration(audio_path)
            analysis_results = predict_emotion_from_image( os.path.join(os.path.dirname(__file__), 'temp/happy.jpg') )
            
            # Display results
            
            st.subheader("Sentiment Analysis on Face Snapshots")
            st.write(analysis_results)
            
            filler_words_csv = ",".join(audio_analysis["filler_word"])

            st.subheader("Speech Analysis")
            st.write(f"Speech analysis: {audio_analysis['wpm']} words per minute")
            st.write(f"You used  {audio_analysis['filler_count']} filler words. Using: {filler_words_csv}")
            st.audio(audio_analysis['audio_path']) 

        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")

