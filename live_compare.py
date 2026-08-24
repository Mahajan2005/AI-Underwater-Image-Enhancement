import cv2
import torch
import numpy as np
from ultralytics import YOLO

from model import UnderwaterEnhancer
from utils import rgb_hsv_fusion

# Device (Mac M-series supported)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load enhancement model
model = UnderwaterEnhancer().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Load YOLO model
yolo = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)

# Create window
cv2.namedWindow("Live Compare", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ================= ENHANCEMENT =================
    frame_resized = cv2.resize(frame, (256, 256))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0

    fused = rgb_hsv_fusion(frame_rgb)
    fused = np.transpose(fused, (2, 0, 1))
    fused = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(fused)

    enhanced = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Normalize
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
    enhanced = (enhanced * 255).astype(np.uint8)

    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    # ================= SHARPEN =================
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # ================= OBJECT DETECTION =================
    results = yolo(enhanced, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])

            # 🔥 FILTER LOW CONFIDENCE
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            label = f"{yolo.names[cls]} {conf:.2f}"

            cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(enhanced, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ================= DISPLAY =================
    original_display = cv2.resize(frame, (480, 270))
    enhanced_display = cv2.resize(enhanced, (480, 270))

    cv2.putText(original_display, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(enhanced_display, "Enhanced + Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # White separator line
    line = np.ones((5, 480, 3), dtype=np.uint8) * 255

    # Stack vertically
    combined = np.vstack((original_display, line, enhanced_display))

    cv2.resizeWindow("Live Compare", 520, 650)
    cv2.imshow("Live Compare", combined)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()