import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pyautogui
import numpy as np
import json
import subprocess
import platform

with open("labels.json", "r") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}
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

model = GestureNet()
model.load_state_dict(torch.load("gesture_model.pth"))
model.eval()


OS = platform.system()
def lock_screen():
    if OS == "Windows":
        subprocess.Popen(["rundll32.exe", "user32.dll,LockWorkStation"])
    elif OS == "Darwin":
        subprocess.Popen(["pmset", "displaysleepnow"])
    else:
        subprocess.Popen(["gnome-screensaver-command", "-l"])


app_map = {
    "peace": ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"],
    "fist": ["notepad.exe"],
    "thumbs_up": ["calc.exe"],
    "open_palm": ["explorer.exe"]
}


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()


alpha_fast, alpha_slow = 0.4, 0.1
v_thresh = screen_w * 0.02
prev_x, prev_y = screen_w//2, screen_h//2

def smooth_pos(target_x, target_y, prev_x, prev_y):
    vel = abs(target_x - prev_x) + abs(target_y - prev_y)
    alpha = alpha_fast if vel > v_thresh else alpha_slow
    cx = prev_x + alpha * (target_x - prev_x)
    cy = prev_y + alpha * (target_y - prev_y)
    return cx, cy

pinch_thresh = 40
zoom_thresh = 120

dragging = False

print("📸 Starting advanced hand control. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)


    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 1:
        preds = []
        for handLms in result.multi_hand_landmarks:
            lm = handLms.landmark
            data = [c for pt in lm for c in (pt.x, pt.y, pt.z)]
            out = model(torch.tensor([data], dtype=torch.float32))
            preds.append(label_map[out.argmax(dim=1).item()])
        if preds.count('open_palm') >= 2:
            lock_screen()
            print("🔒 Screen locked")
        if 'wave' in preds:
            print("👋 Wave detected: dismiss notifications")
        if 'thumbs_down' in preds:
            pyautogui.press('volumemute'); print("🔇 Audio muted")
        if 'okay' in preds:
            pyautogui.press('playpause'); print("⏯️ Play/Pause toggled")

    elif result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
       
        data = [c for pt in lm for c in (pt.x, pt.y, pt.z)]
        with torch.no_grad():
            out = model(torch.tensor([data], dtype=torch.float32))
            gesture = label_map[out.argmax(dim=1).item()]
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

   
        ix, iy = int(lm[8].x*w), int(lm[8].y*h)
        tx, ty = int(lm[4].x*w), int(lm[4].y*h)
        target_x = np.interp(ix, [0,w], [0,screen_w])
        target_y = np.interp(iy, [0,h], [0,screen_h])
        curr_x, curr_y = smooth_pos(target_x, target_y, prev_x, prev_y)
        pyautogui.moveTo(curr_x, curr_y); prev_x, prev_y = curr_x, curr_y

  
        dist = np.hypot(ix-tx, iy-ty)
        if dist < pinch_thresh:
            if not dragging:
                dragging = True
                pyautogui.mouseDown()
            cv2.circle(frame,(ix,iy),12,(0,255,0),-1)
        else:
            if dragging:
                dragging = False
                pyautogui.mouseUp()
            cv2.circle(frame,(ix,iy),12,(0,0,255),2)

        ix2, iy2 = int(lm[12].x*w), int(lm[12].y*h)
        spread = np.hypot(ix-ix2, iy-iy2)
        if spread > zoom_thresh:
            delta = int((spread-zoom_thresh)/5)
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(delta)
            pyautogui.keyUp('ctrl')
            cv2.putText(frame, f"Zoom: {delta}", (10,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    else:
        cv2.putText(frame, "No Hand", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Advanced Hand Control", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows()
