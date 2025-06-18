import cv2
import mediapipe as mp
import numpy as np
import torch
import json


with open("labels.json", "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}


class GestureNet(torch.nn.Module):
    def __init__(self, input_size=63, num_classes=len(label_map)):
        super(GestureNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


model = GestureNet()
model.load_state_dict(torch.load("gesture_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Showing real-time predictions. Press q to quit.")

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(x)
                pred_class = torch.argmax(prediction).item()
                pred_label = label_map[pred_class]

            cv2.putText(frame, f'Gesture: {pred_label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
