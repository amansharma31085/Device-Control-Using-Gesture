🖐️ Device Control Using Gesture
✨ Overview
Welcome to Device Control Using Gesture! This innovative project leverages the power of computer vision and machine learning to enable intuitive, touchless control of your computer using simple hand gestures. Say goodbye to your mouse and keyboard for common tasks – just wave your hand!

Built with Python, OpenCV, MediaPipe, and PyTorch, this system provides a seamless and responsive way to interact with your device in real-time.

🌟 Features
👁️ Real-time Hand Tracking: Utilizes your webcam for robust and accurate hand detection and landmark tracking.

🖱️ Intuitive Mouse Control: Move your mouse cursor smoothly across the screen by simply moving your index finger.

👆 Effortless Clicks: Simulate mouse clicks with a natural pinch gesture (bringing your thumb and index finger together).

🚀 Application Launching: Launch your favorite applications with specific, predefined gestures:

✌️ Peace: Opens Google Chrome

✊ Fist: Launches Notepad

👍 Thumbs Up: Starts Calculator

✋ Open Palm: Opens File Explorer

🔒 System Controls at Your Fingertips:

👐 Double Open Palm: Instantly locks your screen for privacy.

👎 Thumbs Down: Mutes/unmutes your system audio.

👌 OK Gesture: Toggles play/pause for media.

↔️ Two-Finger Spread: Enables zoom in/out functionality (e.g., in browsers or documents).

🧠 Customizable & Adaptable: Easily collect new gesture data and retrain the underlying neural network to recognize your own custom gestures, making the system truly personal.

💪 Robust & Efficient: Powered by the highly optimized MediaPipe for hand detection and a custom GestureNet neural network in PyTorch for precise gesture classification.

⚙️ How It Works
The magic behind this project unfolds through a series of interconnected Python scripts:

collect_data.py:

Purpose: Gathers raw hand landmark data from your webcam for specific gestures.

Process: You provide a gesture name (e.g., "fist", "open_palm"), and the script records the 3D coordinates (x, y, z) of 21 hand landmarks for each frame where a hand is detected. This data is saved as JSON files in the data/ directory.

train_model.py:

Purpose: Trains the GestureNet neural network to recognize the gestures from your collected data.

Process: It loads all JSON data from the data/ directory, prepares it for training, and then trains a deep learning model using PyTorch. The trained model (gesture_model.pth) and a mapping of gesture names to numerical labels (labels.json) are saved.

predict_gesture.py:

Purpose: Provides a real-time visualization of gesture recognition.

Process: Loads the trained model and labels.json, captures video from your webcam, processes hand landmarks in real-time, and overlays the predicted gesture name directly onto the video feed.

gesture_control.py:

Purpose: The main application for real-time device control.

Process: Continuously monitors your hand gestures using the trained model. Based on the recognized gesture and hand position, it uses pyautogui to simulate mouse movements and clicks, and subprocess to launch applications or perform system-level actions (like screen locking or volume control).

retrain.py:

Purpose: Offers an interactive way to expand your gesture library and update the model.

Process: Allows you to record new gesture data on the fly and then retrain the GestureNet model with the augmented dataset, ensuring your system remains adaptable and accurate.

🚀 Getting Started
Prerequisites
Before diving in, make sure you have the following installed on your system:

Python 3.x (recommended 3.8+)

pip (Python package installer)

A functioning webcam

Installation
Clone the repository:

git clone https://github.com/amansharma31085/Device-control-using-gesture.git
cd Device-control-using-gesture

Install dependencies:

pip install opencv-python mediapipe numpy torch pyautogui scikit-learn

Usage
Follow these steps to set up and run the gesture control system:

Collect Data for Your Gestures:
Start by training the system with your own gestures. Run collect_data.py and follow the prompts.

python collect_data.py

Example: When prompted, type fist and make a fist gesture for a few seconds. Repeat for other gestures like open_palm, peace, thumbs_up, thumbs_down, okay, etc.
This will create JSON files (e.g., data/fist.json, data/open_palm.json) in the data/ directory.

Train Your Gesture Recognition Model:
Once you've collected sufficient data for all your desired gestures, train the model:

python train_model.py

This will generate gesture_model.pth (your trained model) and labels.json (the mapping of gesture names to internal labels).

Start Device Control!
With the model trained, you can now launch the main control application:

python gesture_control.py

A webcam feed will appear. Position your hand clearly, and start using your gestures! Press q to quit the application.

Real-time Gesture Prediction (Optional):
If you just want to see the model's predictions in real-time without activating controls, use predict_gesture.py:

python predict_gesture.py

Press q to quit.

Retrain Model (Advanced/Optional):
To add new gestures or improve existing ones, retrain.py offers an interactive way to do so:

python retrain.py

Press r to record a new gesture (you'll be prompted for its name).

Press t to retrain the model with all current data (including newly collected gestures).

Press q to quit.

🤝 Contributing
Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, feel free to open an issue or submit a pull request.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
MediaPipe for robust hand tracking.

OpenCV for webcam integration and image processing.

PyTorch for the deep learning framework.

PyAutoGUI for programmatic control of the mouse and keyboard.
