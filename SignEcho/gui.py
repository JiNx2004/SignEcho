import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
import pyttsx3
import time  # Import time module to add delay

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize GUI
root = tk.Tk()
root.title("Sign Language to Text")

# Create a ScrolledText widget
text_area = scrolledtext.ScrolledText(root, width=50, height=10, font=("Arial", 18))
text_area.pack()

# Function to handle Text-to-Speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to clear the text
def clear_text():
    text_area.delete(1.0, tk.END)

# Video Capture
cap = cv2.VideoCapture(0)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Variable to manage delay between detections
last_detected_time = time.time()
detection_delay = 1.5  # Delay in seconds between detections

# Main Loop
def video_loop():
    global last_detected_time  # Access the global variable
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        return

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the character
        current_time = time.time()
        if current_time - last_detected_time > detection_delay:  # Check delay
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            # Display the prediction
            text_area.insert(tk.END, predicted_character)
            text_area.see(tk.END)  # Scroll to the end

            # Update last detected time
            last_detected_time = current_time

            # Draw the rectangle and put the predicted character on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Show the frame in the OpenCV window
    cv2.imshow('frame', frame)
    
    # Check if the user pressed 'x' to quit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        root.quit()
        cap.release()
        cv2.destroyAllWindows()
    
    # Repeat the loop
    root.after(10, video_loop)

# Function to process the detected text as a word
def process_text():
    text = text_area.get(1.0, tk.END).strip()
    if text:
        speak_text(text)
        clear_text()

# Button to speak the text
speak_button = tk.Button(root, text="Speak Text", command=process_text, font=("Arial", 14))
speak_button.pack(pady=10)

# Start video loop
root.after(0, video_loop)

# Start the GUI main loop
root.mainloop()
