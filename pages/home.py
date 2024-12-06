import streamlit as st
import cv2
import os
import pyaudio
import wave
from threading import Thread

# Will be used as an argument to detect_faces function to show video with the rectangle
# haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def home():
    st.title("Video Display Application")

    # Initialize session state for video toggle
    if "video_visible" not in st.session_state:
        st.session_state.video_visible = False

    # Button to toggle video visibility
    if st.button("Toggle Video"):
        st.session_state.video_visible = not st.session_state.video_visible

    # Display video if the toggle is on
    if st.session_state.video_visible:
        st.write("Facecam is ON")
        run_facecam()
    else:
        st.write("Video is currently hidden. Click the button to view it.")







# Helper functions

def initialize_face_cascade():
    """
    Load the Haar Cascade for face detection.
    Returns:
        haar_cascade (cv2.CascadeClassifier): The pre-trained face detection model.
    """
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



def detect_faces(frame, haar_cascade):
    """
    Detect faces in the given frame using Haar Cascade and only consider the largest face.
    Args:
        frame (ndarray): The image frame from the webcam.
        haar_cascade (cv2.CascadeClassifier): The face detection model.

    Returns:
        frame (ndarray): The frame with the largest face marked.
        face (tuple): The rectangle of the largest face detected (x, y, w, h).
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return frame, None  # No faces detected, return the original frame

    # Find the largest face by area (width * height)
    largest_face = max(faces, key=lambda x: x[2] * x[3])  # x[2] is width, x[3] is height

    # Draw a rectangle around the largest face
    (x, y, w, h) = largest_face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle around largest face

    return frame, largest_face



def display_frame(frame, frame_placeholder):
    """
    Display the processed frame in the Streamlit placeholder.
    Args:
        frame (ndarray): The processed frame.
        frame_placeholder (st.empty): Streamlit placeholder to show the frame.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    frame = cv2.flip(frame, 1)  # Flip frame horizontally (mirror)
    frame_placeholder.image(frame, channels="RGB")  # Display in the placeholder



def record_audio(filename):
    """ Record audio from the microphone and save to a file. """
    p = pyaudio.PyAudio()

    # Set up audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)
    
    frames = []
    for _ in range(0, int(44100 / 1024 * 20)):  # Record for 20 seconds (adjust as needed)
        data = stream.read(1024)
        frames.append(data)
    
    # Stop the stream and save the audio file
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))



def run_facecam():
    """
    Main function to capture webcam video and display it with face detection.
    """
    cap = cv2.VideoCapture(0)  # Access the system's webcam (0 for default webcam)
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
        return
    

    haar_cascade = initialize_face_cascade()  # Load Haar Cascade for face detection
    frame_placeholder = st.empty()  # Placeholder for live frames

    # Session state for recording toggle
    if "recording" not in st.session_state:
        st.session_state.recording = False  # Initially not recording
        st.session_state.video_writer = None  # No video writer initially

    # Button for toggling video recording
    record_button = st.button("Start/Stop Recording")

    if record_button:
        if st.session_state.recording:
            # Stop recording
            st.session_state.recording = False
            if st.session_state.video_writer:
                st.session_state.video_writer.release()  # Release the video writer
                st.session_state.video_writer = None  # Reset the writer
            
            # Stop audio recording
            if st.session_state.audio_thread:
                st.session_state.audio_thread.join()  # Ensure the audio thread finishes
            st.success("Recording stopped. Traverse to the result page from the sidebar")
            
        else:
            # Start recording
            st.session_state.recording = True
            # Define the temp folder path relative to home.py's location
            temp_folder = os.path.join(os.path.dirname(__file__), 'temp')  # Path to 'temp' folder
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)  # Create 'temp' folder if it doesn't exist

            video_path = os.path.join(temp_folder, "recorded_video.mp4")  # Save video as .mp4
            audio_path = os.path.join(temp_folder, "recorded_audio.wav")  # Save audio as .wav

            # Get the frame dimensions from the webcam capture
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to capture the frame from the webcam.")
                return
            height, width, _ = frame.shape  # Get the frame height and width
            
            # get_actual_fps() function always returns fallback - using explicitly defined fps
            dynamic_fps = 7

            # Create a VideoWriter object to save the video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for .mp4
            out = cv2.VideoWriter(video_path, fourcc, dynamic_fps, (width, height))  # Use captured frame size
            st.session_state.video_writer = out  # Store the writer in session state

            # Start audio recording in a separate thread
            audio_thread = Thread(target=record_audio, args=(audio_path,))
            audio_thread.start()
            st.session_state.audio_thread = audio_thread  # Store the thread in session state
            st.success(f"Recording started. Saving to: {video_path} and {audio_path}")

    while st.session_state.video_visible:
        ret, frame = cap.read()  # Capture frame from the webcam

        if not ret:
            st.error("Unable to capture the frame from the webcam.")
            break

        # Detect faces in the frame
        frame, faces = detect_faces(frame, haar_cascade)

        # Display the frame with face detection
        display_frame(frame, frame_placeholder)

        # Record the frame if recording is active
        if st.session_state.recording and st.session_state.video_writer:
            st.session_state.video_writer.write(frame)  # Write frame to video file

        # Stop streaming when the button is toggled off
        if not st.session_state.video_visible:
            break

    # Release resources and cleanup
    cap.release()
    cv2.destroyAllWindows()
    frame_placeholder.empty()  # Clear the placeholder after stopping the feed



# Run the app if this file is executed
if __name__ == "__main__":
    home()