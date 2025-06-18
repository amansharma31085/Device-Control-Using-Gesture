import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split


data = []
labels = []
label_map = {}
label_index = 0

for file in os.listdir("data"):
    if file.endswith(".json"):
        with open(os.path.join("data", file)) as f:
            samples = json.load(f)
            for sample in samples:
                data.append(sample["landmarks"])
                labels.append(label_index)
        label_map[label_index] = file.replace(".json", "")
        label_index += 1

print(f"Loaded {len(data)} samples from {len(label_map)} gestures.")


X = np.array(data, dtype=np.float32)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_X = torch.tensor(X_train)
train_y = torch.tensor(y_train, dtype=torch.long)
test_X = torch.tensor(X_test)
test_y = torch.tensor(y_test, dtype=torch.long)

class GestureNet(nn.Module):
    def __init__(self, input_size=63, num_classes=len(label_map)):
        super(GestureNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

model = GestureNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    outputs = model(train_X)
    loss = criterion(outputs, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(test_X).argmax(dim=1)
        acc = (preds == test_y).float().mean()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, Test Acc: {acc.item()*100:.2f}%")

torch.save(model.state_dict(), "gesture_model.pth")
with open("labels.json", "w") as f:
    json.dump(label_map, f)

print("Training complete.")
