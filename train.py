import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np

from model import UnderwaterEnhancer
from utils import rgb_hsv_fusion

# Device (Mac M4)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = UnderwaterEnhancer().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_folder = "dataset/input"
target_folder = "dataset/target"

epochs = 25

for epoch in range(epochs):
    total_loss = 0

    for img_name in os.listdir(input_folder):

        inp_path = os.path.join(input_folder, img_name)
        tgt_path = os.path.join(target_folder, img_name)

        inp = cv2.imread(inp_path)
        tgt = cv2.imread(tgt_path)

        inp = cv2.resize(inp, (256, 256))
        tgt = cv2.resize(tgt, (256, 256))

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB) / 255.0
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB) / 255.0

        fused = rgb_hsv_fusion(inp)
        fused = np.transpose(fused, (2, 0, 1))

        inp_tensor = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(device)
        tgt_tensor = torch.tensor(np.transpose(tgt, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

        output = model(inp_tensor)
        loss = criterion(output, tgt_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")