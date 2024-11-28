
# Contains helper functions


import speech_recognition as sr
import cv2
import numpy as np
import streamlit as st 
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('models/TedEmotion_model.keras')

# Initialize recognizer
r = sr.Recognizer()

# Detect filler words
filler_words = [
    "um", "uh", "like", "so", "you know", "actually", "basically", "literally",
    "totally", "really", "just", "well", "kind of", "sort of", "i mean",
    "right", "okay", "alright", "anyway", "hmm"
]

# Class indices as per your train_generator
class_indices = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 
                 'sad': 5, 'surprise': 6}

# Reverse the class indices for decoding predictions
class_labels = {v: k for k, v in class_indices.items()}


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

        # Display the face region on Streamlit UI
        # st.image(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB), caption=f"Face at {round(frame_time, 2)}s", use_column_width=True)

        # Resize the face region to 48x48
        resized_face = cv2.resize(face_region, (48, 48))

        # Convert the resized face to grayscale (1 channel)
        gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
        
        # Normalize the resized face
        normalized_face = gray_face / 255.0  # Normalize pixel values
        input_face = np.expand_dims(normalized_face, axis=(0, -1))  # Add batch and channel dimensions
        
        # Predict the raw emotion probabilities for the cropped face
        predictions = model.predict(input_face)[0]  # Extract predictions for the single frame
        
        # Map raw predictions to class labels (no normalization applied)
        prediction_dict = {class_labels[i]: float(pred) for i, pred in enumerate(predictions)}

        # Store the raw result
        analysis_results[round(frame_time, 2)] = prediction_dict

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




def predict_emotion_from_image(image_path):
    """
    Load the image, preprocess it, and predict the emotion using the trained model.
    Args:
        image_path (str): Path to the image file (e.g., 'temp/happy.jpg').
    Returns:
        dict: Dictionary with emotions as keys and predicted values as values.
    """
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize the image to 48x48 (the input size for your model)
    resized_img = cv2.resize(img, (48, 48))
    
    # Convert the image to grayscale (if your model expects grayscale input)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to [0, 1]
    normalized_img = gray_img / 255.0
    
    # Add batch and channel dimensions
    input_img = np.expand_dims(normalized_img, axis=(0, -1))  # (1, 48, 48, 1)

    # Predict the emotion using the loaded model
    predictions = model.predict(input_img)[0]  # Get the first (and only) prediction
    
    # Map predictions to class labels
    prediction_dict = {class_labels[i]: float(pred) for i, pred in enumerate(predictions)}
    
    # Display image and prediction
    st.image(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), caption="Predicted Image", use_column_width=True)
    
    # Display the raw prediction
    st.write("Prediction:", prediction_dict)
    
    return prediction_dict