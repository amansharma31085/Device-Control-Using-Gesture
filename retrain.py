import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
import pyautogui
import numpy as np
import json
import subprocess
import platform
import os
import time
from sklearn.model_selection import train_test_split

MODEL_PATH = "gesture_model.pth"
LABELS_PATH = "labels.json"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
else:
    label_map = {}
num_classes = len(label_map)

class GestureNet(nn.Module):
    def __init__(self, input_size=63, num_classes=num_classes):
        super(GestureNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.fc(x)

def save_labels_map():
    inv = {v: k for k, v in label_map.items()}
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)

def collect_for_gesture(label_name, duration=5):
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    frames = []
    start = time.time()
    print(f"Recording '{label_name}' for {duration}s...")
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            arr = [c for pt in lm for c in (pt.x, pt.y, pt.z)]
            frames.append(arr)
        cv2.imshow("Collecting Data...", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyWindow("Collecting Data...")

    path = os.path.join(DATA_DIR, f"{label_name}.json")
    with open(path, "w") as f:
        json.dump([{"label": label_name, "landmarks": f} for f in frames], f)
    print(f"Saved {len(frames)} samples to {path}")


def retrain_model(epochs=10):
    data, labels = [], []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.json'):
            label = fname.replace('.json','')
            if label not in label_map.values():
                idx = max(label_map.keys(), default=-1) + 1
                label_map[idx] = label
            idx = next(k for k,v in label_map.items() if v==label)
            with open(os.path.join(DATA_DIR,fname)) as f:
                samples = json.load(f)
                for s in samples:
                    data.append(s['landmarks']); labels.append(idx)
    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    global model, num_classes
    num_classes = len(label_map)
    model = GestureNet(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
  
    for e in range(epochs):
        model.train()
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        acc = (model(X_test).argmax(1)==y_test).float().mean().item()
        print(f"Epoch {e+1}/{epochs}: loss={loss.item():.4f}, acc={acc:.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    save_labels_map()
    print("Retraining complete and model saved.")


model = GestureNet(num_classes=num_classes)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
print("Press 'r' to record new gesture, 't' to retrain, 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        arr = [c for pt in lm for c in (pt.x,pt.y,pt.z)]
        with torch.no_grad():
            out = model(torch.tensor([arr],dtype=torch.float32))
            pred = out.argmax(1).item()
            name = label_map.get(pred,'')
        cv2.putText(frame,name,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Gesture Control",frame)
    key = cv2.waitKey(1)&0xFF
    if key==ord('r'):
        lbl = input("Enter new gesture name: ")
        collect_for_gesture(lbl)
    elif key==ord('t'):
        retrain_model()
    elif key==ord('q'):
        break
cap.release(); 
cv2.destroyAllWindows()
