import cv2
import numpy as np
import io
import PIL
from base64 import b64decode, b64encode
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('emotion_model.h5',compile=False)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Define functions to convert between JavaScript image reply and OpenCV image
def js_to_image(js_reply):
    image_bytes = b64decode(js_reply.split(',')[1])
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    return img

def bbox_to_bytes(bbox_array):
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
    iobuf = io.BytesIO()
    bbox_PIL.save(iobuf, format='png')
    bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))
    return bbox_bytes

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


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_region, (48, 48))
            img_array = np.expand_dims(face_resized, axis=0)
            img_array = np.expand_dims(img_array, axis=-1)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_emotion = emotion_dict[predicted_class]
            accuracy = predictions[0][predicted_class]
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"{predicted_emotion} ({accuracy:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame.from_ndarray(img, format="bgr24")





# Page Title and Description
st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")
st.title("Facial Emotion Recognition")

# Sidebar
st.sidebar.title("Options")
option = st.sidebar.radio("Select Option", ("Drag a File","Process Video"))

# Main Content Area
if option == "Drag a File" :
    st.subheader("Photo Processing")
    
    # Process image or captured frame
    if option == "Drag a File":
        uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
    
    if 'image' in locals():
        processed_frame, emotions = process_frame(image)
        # Display processed frame and emotions
        st.subheader("Processed Frame")
        st.image(processed_frame, channels="BGR", use_column_width=False)
        if not emotions:
            st.warning("No faces detected in the image.")
elif option == "Process Video":
    webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)


    
