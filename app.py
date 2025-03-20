import numpy as np
import cv2
import streamlit as st
from keras.models import load_model
import streamlit_webrtc as webrtc
from PIL import Image

# Load the model and emotion dictionary
model = load_model('emotion_model.h5')
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process frame and detect emotions
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (48, 48))
        img = np.expand_dims(face_resized, axis=0)
        img = np.expand_dims(img, axis=-1)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        predicted_emotion = emotion_dict[predicted_class]
        accuracy = predictions[0][predicted_class]

        # Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{predicted_emotion} ({accuracy:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return frame

# Define WebRTC video stream handler
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed_frame = process_frame(img)
    return webrtc.VideoFrame.from_ndarray(processed_frame, format="bgr24")

# WebRTC configuration
st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")
st.title("Facial Emotion Recognition")

# Streamlit WebRTC Video Stream
st.subheader("Video Stream")
webrtc.webrtc_streamer(key="emotion-recognition", video_frame_callback=video_frame_callback)
