
# Contains helper functions


import speech_recognition as sr
import cv2
import numpy as np
import streamlit as st 
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('/Users/ayush/Documents/Stuff/ML Boi/Project/oratorAI/final/models/best_model_2.keras')

# Initialize recognizer
r = sr.Recognizer()

# Detect filler words
filler_words = [
    "um", "uh", "like", "so", "you know", "actually", "basically", "literally",
    "totally", "really", "just", "well", "kind of", "sort of", "i mean",
    "right", "okay", "alright", "anyway", "hmm"
]

# Class indices as per your train_generator
class_indices = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 
                 5: 'sad',6: 'surprise',-1: 'others'}


# Dummy function for transcript extraction - will be implemented in next push
def extract_transcript(audiofile_path):
    """
    Extract transcript from audio for predefined trait analysis.
    Args:
        audio_path (str): Path to the saved audio file.
    Returns:
        str: Dummy transcript text.
    """
    # TODO: Implement actual audio-to-text logic later - Thinking of leaving this part of project @@
    return 



def extract_audio_and_duration(audiofile_path):
# Dummy function for Speech analysis
    """
    Extract the audio duration and audio for cadence analysis.
    Args:
        audio_path (str): Path to the saved audio file.
    Returns:
        dict: speaking speed, filler words count and audio path.
    """
    
    # Capture audio - will properly module the code later @@
    with sr.AudioFile(audiofile_path) as source:
        audio = r.record(source)

    # Convert audio to text - Google Web Speech API.
    text = r.recognize_google(audio)
    
    # Analyze speaking duration
    words = text.split()

    # Number of audio frames divided by the sample rate
    speech_duration = len(audio.frame_data) / audio.sample_rate
    speaking_duration = ( len(words) / speech_duration ) * 60

    filler_words_used = [word for word in filler_words if word in text]
    filler_count = len(filler_words_used)
    
    return {
        "wpm": int(speaking_duration), 
        "filler_count": str(filler_count), 
        "filler_word": filler_words_used, 
        "audio_path": audiofile_path
        }


def extract_snapshots_for_sentiment(video_path):
    """
    Extract snapshots from the video every 1/8th of its duration, perform sentiment analysis on each frame,
    and return the results in a dictionary with raw sentiment prediction values.
    Args:
        video_path (str): Path to the saved video file.
    Returns:
        dict: Dictionary with time points as keys and raw sentiment predictions as values.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # in seconds
    interval = duration / 8  # Snapshot interval (1/8th of video duration)

    # Dictionary to store analysis results
    analysis_results = {}

    # Iterate through the video and extract frames
    for i in range(8):
        # Calculate the frame to extract
        frame_time = interval * i
        frame_id = int(frame_time * fps)

        # Set video to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            print(f"Could not read frame at time {frame_time}s.")
            continue

        # Get the largest face detected in the frame
        face_bbox = detect_largest_face(frame)

        if face_bbox is None:
            print(f"No face detected at {frame_time}s.")
            continue  # Skip if no face is detected
        
        # Get the coordinates of the largest face
        (x, y, w, h) = face_bbox

        # Crop the face region from the full frame
        face_region = frame[y:y+h, x:x+w]


    

        # Resize the face region to 48x48
        resized_face = cv2.resize(face_region, (48, 48))

        # Convert the resized face to grayscale (1 channel)
        gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
        
        # For testing - visualing face
        # st.image(gray_face, caption="Processed Face (Grayscale)", use_container_width=True)

        # Normalize the resized face
        normalized_face = gray_face / 255.0  # Normalize pixel values
        input_face = np.expand_dims(normalized_face, axis=(0, -1))  # Add batch and channel dimensions
        
        

        # Predict the raw emotion probabilities for the cropped face
        predictions = model.predict(input_face)[0]  # Extract predictions for the single frame

        # Extracting top 2 indices
        top_2_indices = np.argsort(predictions)[-2:]

        # Dictionary with scores of top 2 emotions with their index
        sortedPred = {}
        for index in top_2_indices:
            sortedPred[index] = predictions[index]
        
        # Append the sum of remaining values (rest of the emotions)
        sortedPred[-1] = 1 - sum(sortedPred.values())

        # Encoding the values with the emotion name
        sortedPred = {class_indices[key]: value for key,value in sortedPred.items()}

        # Normalize the values to avoid floating point errors
        total = sum(sortedPred.values())
        normalizedPred = {key: round(value / total, 3) for key, value in sortedPred.items()}

        # Store the result for the current frame in the big dictionary
        analysis_results[round(frame_time, 2)] = normalizedPred

    # Release video capture
    cap.release()
    return analysis_results


def detect_largest_face(frame):
    """
    Detects faces in a frame and returns the bounding box of the largest face.
    Args:
        frame (ndarray): The frame in which to detect faces.
    Returns:
        tuple: Coordinates (x, y, w, h) of the bounding box of the largest face.
    """
    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # No face detected

    # Find the largest face by area (w * h)
    largest_face = max(faces, key=lambda face: face[2] * face[3])  # face[2] = width, face[3] = height

    return largest_face  # (x, y, w, h)
