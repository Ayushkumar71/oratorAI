import os

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from helper import extract_snapshots_for_sentiment, extract_audio_and_duration



def results():
    st.title("Analysis Results")
    
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
            analysis_results = extract_snapshots_for_sentiment(video_path)
            audio_analysis = extract_audio_and_duration(audio_path)


            # Video(visual) analysis
            st.subheader("Sentiment Analysis on Face Snapshots")
            
            plot_stacked_area_chart(analysis_results)
            st.markdown("""*Note: The blend of emotions shown is inferred from probability distributions 
                        and may not reflect exact ground truth. Interpret results with caution.*""")
            


            # Audio Analysis
            filler_words_csv = ", ".join(audio_analysis["filler_word"])
            wpm = audio_analysis['wpm']

            st.subheader("Speech Analysis")

            # Common text 
            st.write(f"""Speech analysis: {wpm} words per minute. You 
                     used  {audio_analysis['filler_count']} filler words. Using : {filler_words_csv}""")
            
            # Conditional text based on wpm
            if wpm < 60:
                st.write(f"Your speech might feel too slow; consider speaking a bit faster for better engagement.")
            elif 60 <= wpm <= 110:
                st.write(f"This is a moderate pace, suitable for most audiences.")
            elif 111 <= wpm <= 150:
                st.write(f"This is a slightly fast pace; it might work well for dynamic presentations.")
            else:
                st.write(f"Your speech is very fast; consider slowing down to ensure clarity for your audience.")
            
            st.success("Move to help page for helpfull tips.")
        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")


# Helper function - For plotting area chart
def plot_stacked_area_chart(analysis_results):
    """
    Function to plot a stacked area chart showing the evolution of class probabilities over time.

    Parameters:
    analysis_results (dict): Dictionary where keys are timestamps and values are dictionaries of class probabilities.
    """
    # Prepare the data for visualization
    frames = []
    classes = []

    for time, class_probs in analysis_results.items():
        for class_index, prob in class_probs.items():
            frames.append(time)
            classes.append({"Time": time, "Class": class_index, "Probability": prob})

    # Convert to DataFrame
    df = pd.DataFrame(classes)

    # Pivot the DataFrame to have Classes as columns
    df_pivot = df.pivot(index="Time", columns="Class", values="Probability").fillna(0)
    df_pivot = df_pivot.sort_index()  # Ensure frames are sorted by time

    # Plot the stacked area chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(df_pivot.index, df_pivot.T, labels=df_pivot.columns, alpha=0.7)

    # Add legend and labels
    ax.legend(title="Classes", loc="upper left", bbox_to_anchor=(1, 1))
    plt.title("Class Probabilities Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability")
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)