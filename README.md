# Facial Emotion Recognition System

## Project Overview
you can consult the app on this link:
https://facial-emotion-recognition-system.streamlit.app/

The **Facial Emotion Recognition System** is a deep learning-based application that identifies human emotions by analyzing facial expressions in images or real-time video streams. This project uses a Convolutional Neural Network (CNN) for emotion detection and has been implemented with **Streamlit** to provide an intuitive and user-friendly interface for interaction. Users can upload images or use their webcam to detect emotions in real time.

![image](https://github.com/user-attachments/assets/143ffaba-b908-4b5b-962b-c53530015e3f)

---

## Features

- **Emotion Detection from Images**: Users can upload an image, and the system will analyze the facial expressions to identify emotions.
- **Real-Time Emotion Detection via Webcam**: The system captures real-time video feed from the user's webcam, processes it frame-by-frame, and predicts emotions.
- **Streamlit Interface**: A clean and responsive web interface for interacting with the system, designed for simplicity and ease of use.
- **Emotion Categories**: The system can detect and classify multiple emotions, such as:
  - Happy
  - Sad
  - Angry
  - Surprised
  - Neutral
  - Fearful
  - Disgusted

---

## How It Works

1. **Image Preprocessing**: 
   - Uploaded images or webcam frames are preprocessed, including resizing and grayscale conversion, to match the input requirements of the CNN model.
   - Facial detection is applied to isolate faces in the image.

2. **Emotion Prediction**:
   - The preprocessed facial images are passed through a pre-trained CNN model to predict the emotion.
   - The model outputs a probability distribution over the possible emotions, and the emotion with the highest probability is selected as the prediction.

3. **Result Display**:
   - For images: The detected emotions are displayed along with the processed image.
   - For webcam: Real-time predictions are overlayed on the video stream.

---

## Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras
  - Developed a CNN model trained on a labeled dataset of facial expressions.
    ![model_plot](https://github.com/user-attachments/assets/33eb4dd3-0349-4072-bef2-838cb490ff90)

- **Frontend Framework**: Streamlit
  - Provides a web-based interface for interaction with the system.
- **Libraries**:
  - **OpenCV**: For real-time video capture and facial detection.
  - **NumPy & Pandas**: For data manipulation.
  - **Matplotlib**: For visualizing results (if required).
- **Deployment**: The project can be deployed locally or on cloud platforms for accessibility.

---

## Setup and Installation

### Prerequisites
- Python 3.7 or above
- Required libraries: `tensorflow`, `streamlit`, `opencv-python`, `numpy`, `pandas`, `matplotlib`

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/MISSAOUI-MOHAMED-AMINE/Facial-emotion-Recognition.git
   cd Facial-Emotion-Recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Access the application in your browser at `http://localhost:8501`.

---

## Dataset

The CNN model is trained on a publicly available dataset of labeled facial expressions. The dataset contains diverse images representing various emotions to ensure robust and accurate predictions.


---

## Future Enhancements

- Add support for detecting multiple faces in a single frame.
- Improve real-time performance and reduce latency for video processing.
- Expand the emotion categories and fine-tune the model for better accuracy.
- Deploy the application as a public web service or mobile app.

---

## Contributing

Contributions are welcome! If you have ideas for improvement or new features, feel free to create a pull request or open an issue.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Special thanks to the creators of the dataset and the open-source community for the tools and frameworks used in this project.
