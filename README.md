🤖 Device Control Using Gesture
This project is an AI-powered computer vision system that allows you to control your device using hand gestures. Built with OpenCV, MediaPipe, and Python, it enables touchless interaction by recognizing hand and finger movements to perform real-time actions such as mouse movement, clicks, and launching applications.

📌 Features
Real-time Hand Tracking: Utilizes your webcam for accurate, real-time hand tracking.

Mouse Control: Move the mouse cursor smoothly using your index finger.

Click Simulation: Perform clicks using a pinch gesture (thumb + index finger).

Application Launching: Launch specific applications like Notepad, Calculator, Chrome, and File Explorer with predefined gestures (e.g., fist for Notepad, thumbs_up for Calculator, peace for Chrome, open_palm for File Explorer).

System Controls:

Screen Lock: Lock your screen with an open_palm gesture using two hands.

Volume Mute: Mute audio with a thumbs_down gesture.

Play/Pause Toggle: Toggle media play/pause with an okay gesture.

Zoom Control: Zoom in/out by spreading two fingers.

Customizable Gestures: Easily collect new gesture data and retrain the model to recognize custom gestures.

Robust Framework: Built on the powerful MediaPipe and OpenCV frameworks for hand tracking and computer vision, with a custom GestureNet neural network implemented in PyTorch for gesture recognition.

⚙️ How it Works
The system operates in several key stages:

Data Collection (collect_data.py): This script allows you to record hand landmark data for specific gestures using your webcam. It prompts you to enter a gesture name (e.g., "fist", "open", "peace") and saves the collected landmarks as JSON files in a data/ directory. Each sample includes the gesture label and 3D coordinates (x, y, z) for 21 hand landmarks.

Model Training (train_model.py): After collecting data, this script trains a neural network, GestureNet, using the collected landmark data. It splits the data into training and testing sets, and then trains the model to classify gestures. The trained model (gesture_model.pth) and a mapping of labels to indices (labels.json) are saved for later use.

Real-time Prediction (predict_gesture.py): This script loads the trained model and provides real-time gesture recognition. It captures video from your webcam, processes hand landmarks, and displays the predicted gesture label on the screen.

Device Control (gesture_control.py): This is the core application for controlling your device. It loads the trained GestureNet model and uses MediaPipe to detect hands in real-time. Based on the recognized gestures and hand movements, it utilizes pyautogui and subprocess to simulate mouse actions (movement, clicks), launch applications, and perform system-level controls like screen locking, volume muting, and media play/pause.

Retraining (retrain.py): This script offers an interactive way to collect new gesture data and retrain the model directly. It allows you to record new gestures and then retrain the GestureNet model with the updated dataset, making the system adaptable to new gestures and environments.

🚀 Getting Started
Prerequisites
Before you begin, ensure you have the following installed:

Python 3.x

pip (Python package installer)

A webcam

Installation
Clone the repository:

Bash

git clone https://github.com/amansharma31085/Device-control-using-gesture.git
cd Device-control-using-gesture
Install dependencies:

Bash

pip install opencv-python mediapipe numpy torch pyautogui scikit-learn
Usage
Collect Data for Gestures:
Run collect_data.py to gather data for your desired gestures. Follow the on-screen prompts to name your gestures (e.g., fist, open, peace).

Bash

python collect_data.py
This will create JSON files in a data/ directory (e.g., data/fist.json, data/open.json).

Train the Model:
Once you have collected enough data, run train_model.py to train your gesture recognition model. This will generate gesture_model.pth and labels.json.

Bash

python train_model.py
Run Device Control:
After training, execute gesture_control.py to start controlling your device with gestures.

Bash

python gesture_control.py
Ensure your webcam is active and visible. Press 'q' to quit.

Real-time Gesture Prediction (Optional):
To see real-time predictions without device control, use predict_gesture.py.

Bash

python predict_gesture.py
Press 'q' to quit.

Retrain Model (Optional):
Use retrain.py to collect new gesture data and retrain the model directly from an interactive interface.

Bash

python retrain.py
Press 'r' to record a new gesture, 't' to retrain the model, and 'q' to quit.
