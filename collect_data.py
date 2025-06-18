import cv2
import mediapipe as mp
import numpy as np
import json
import os

# === Setup ===
os.makedirs("data", exist_ok=True)
gesture_name = input("Enter gesture name (e.g. fist, open, peace): ")
data = []

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# === Choose Webcam Index (0 or 1 or 2 depending on your system) ===
CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print(f"❌ Could not open webcam at index {CAM_INDEX}")
    exit()

print("Collecting data... Press 'q' to stop.")

try:
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("❌ Failed to read from camera.")
            continue

        frame = cv2.flip(frame, 1)  # Mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        result = hands.process(rgb)

        # Show raw webcam feed regardless
        cv2.imshow("Webcam Feed", frame)

        if result.multi_hand_landmarks:
            print("✅ Hand detected.")
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                data.append({
                    "label": gesture_name,
                    "landmarks": landmarks
                })

            # Optional: Show with landmarks overlaid
            cv2.imshow("Gesture with Landmarks", frame)
        else:
            print("🟡 No hand detected in this frame.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Save only if data was collected
    if data:
        save_path = f"data/{gesture_name}.json"
        with open(save_path, "w") as f:
            json.dump(data, f)
        print(f"✅ Saved {len(data)} samples of '{gesture_name}' to {save_path}")
    else:
        print(f"⚠️ No samples were collected for '{gesture_name}'. Try adjusting lighting, camera, or position.")
