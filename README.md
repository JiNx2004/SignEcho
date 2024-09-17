# SignEcho

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


# Introduction
The Deaf and hard-of-hearing live with sign language as their primary means of communication. Generally speaking, the use of sign language by some people further aggravates the communication gap between them and 
the rest of the world. This project takes a big step toward closing that divide with its ability to translate hand signals into spoken words using computer vision and machine learning techniques.
This system operates by capturing hand gestures through a webcam in real-time, interpreting them with built-in hand tracking algorithms, then displaying the corresponding text and producing an audio output.

# Objective
The primary objective of this project is to develop an interactive system that recognizes sign language gestures in real-time and translates them into audible speech. This is achieved by:

->Capturing hand gestures via webcam and accurately detecting hand landmarks.< br / >
->Classifying these gestures using a trained machine learning model.< br / >
->Converting the recognized signs into spoken language using a text-to-speech engine.< br / >
->Providing a user-friendly interface that facilitates real-time interaction between sign language users and the system.< br / >

# Running the project
1.Clone the repository.< br / >
2.Install the required dependencies:< br / >
  pip install opencv-python mediapipe scikit-learn pyttsx3 matplotlib numpy< br / >
3.Run the data collection script to capture images for each gesture.< br / >
4.Use the preprocessing script to extract landmarks and prepare the dataset.< br / >
5.Train the model using the model training script.< br / >
6.Run the real-time recognition script with the GUI to start detecting and speaking the recognized< br / >



