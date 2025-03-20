import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('emotion_model.h5')
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Define function to process each frame from the video stream
def process_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray)

    emotions = []

    # Process each detected face
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (48, 48))
        img = np.expand_dims(face_resized, axis=0)
        img = np.expand_dims(img, axis=-1)
        predictions = model.predict(img)
        emo = model.predict(img)[0]
        emotions.append(emo)
        predicted_class = np.argmax(predictions)
        predicted_emotion = emotion_dict[predicted_class]
        accuracy = predictions[0][predicted_class]

        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{predicted_emotion} ({accuracy:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return frame, emotions

# Page Title and Description
st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")
st.title("Facial Emotion Recognition")

# Main Content Area
run = st.button('Start Webcam')
stop = st.button('Stop')

if run:
    # Start the webcam feed
    camera = cv2.VideoCapture(0)

    # Display the webcam feed in Streamlit
    FRAME_WINDOW = st.image([])

    while True:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, emotions = process_frame(frame)

        # Display processed frame
        FRAME_WINDOW.image(processed_frame, channels="RGB", use_container_width=True)

        if stop:
            st.write('Stopped')
            break

    camera.release()
