import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import time

from model import UnderwaterEnhancer
from utils import rgb_hsv_fusion

# ---------------- DEVICE ----------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = UnderwaterEnhancer().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture("underwater.mp4")

cv2.namedWindow("AI Underwater Enhancement Dashboard", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Underwater Enhancement Dashboard", 1400, 950)

# ---------------- PSNR ----------------
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

# ---------------- ROUNDED RECT ----------------
def rounded_rect(img, pt1, pt2, color, thickness=-1, r=18):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)

    cv2.circle(img, (x1+r, y1+r), r, color, thickness)
    cv2.circle(img, (x2-r, y1+r), r, color, thickness)
    cv2.circle(img, (x1+r, y2-r), r, color, thickness)
    cv2.circle(img, (x2-r, y2-r), r, color, thickness)

# ---------------- LIVE GRAPH ----------------
def draw_graph(canvas, values, title, color, x, y, w, h):

    rounded_rect(canvas, (x, y), (x+w, y+h), (25,25,25), -1)

    cv2.putText(canvas, title, (x+15, y+30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

    if len(values) < 2:
        return

    vals = values[-60:]

    # Auto zoom
    minv = min(vals)
    maxv = max(vals)

    if abs(maxv-minv) < 0.001:
        maxv += 0.001
        minv -= 0.001

    pad = (maxv-minv)*0.2
    minv -= pad
    maxv += pad

    # Axis
    cv2.line(canvas, (x+55, y+h-35), (x+w-15, y+h-35), (70,70,70), 1)
    cv2.line(canvas, (x+55, y+45), (x+55, y+h-35), (70,70,70), 1)

    # Grid
    for i in range(1,4):
        gy = int(y+45 + i*((h-80)/4))
        cv2.line(canvas, (x+55, gy), (x+w-15, gy), (40,40,40), 1)

    # Labels
    cv2.putText(canvas, f"{maxv:.3f}", (x+5, y+55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    cv2.putText(canvas, f"{minv:.3f}", (x+5, y+h-35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    # Plot points
    pts = []

    for i, v in enumerate(vals):
        px = int(x + 55 + i*((w-75)/(len(vals)-1)))
        norm = (v-minv)/(maxv-minv)
        py = int(y+h-35 - norm*(h-80))
        pts.append((px, py))

    # Glow line
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], color, 5)
        cv2.line(canvas, pts[i-1], pts[i], color, 2)

    # Last point
    cv2.circle(canvas, pts[-1], 6, (255,255,255), -1)
    cv2.circle(canvas, pts[-1], 4, color, -1)

    cv2.putText(canvas, f"{vals[-1]:.3f}",
                (pts[-1][0]-20, pts[-1][1]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

# ---------------- VARIABLES ----------------
frame_count = 0
psnr_total = 0
ssim_total = 0

psnr_history = []
ssim_history = []

prev_time = time.time()

# ---------------- LOOP ----------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # FPS
    current = time.time()
    fps = 1 / (current - prev_time)
    prev_time = current

    # Resize
    frame_resized = cv2.resize(frame, (256,256))

    # RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0

    # Fusion
    fused = rgb_hsv_fusion(frame_rgb)
    fused = np.transpose(fused, (2,0,1))
    fused = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(fused)

    enhanced = output.squeeze(0).permute(1,2,0).cpu().numpy()
    enhanced = (enhanced * 255).clip(0,255).astype(np.uint8)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    # Metrics
    gray1 = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    psnr_value = calculate_psnr(frame_resized, enhanced)
    ssim_value = ssim(gray1, gray2)

    psnr_total += psnr_value
    ssim_total += ssim_value

    avg_psnr = psnr_total / frame_count
    avg_ssim = ssim_total / frame_count

    psnr_history.append(psnr_value)
    ssim_history.append(ssim_value)

    # ---------------- TOP VIDEO ----------------
    original = cv2.resize(frame_resized, (680,360))
    enhanced_view = cv2.resize(enhanced, (680,360))

    cv2.putText(original, "Original Feed", (20,35),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,120), 2)

    cv2.putText(enhanced_view, "Enhanced Feed", (20,35),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0,165,255), 2)

    top = np.hstack((original, enhanced_view))

    # ---------------- DASHBOARD ----------------
    dash = np.zeros((560,1360,3), dtype=np.uint8)
    dash[:] = (10,10,10)

    cv2.putText(dash,
                "AI UNDERWATER ENHANCEMENT CONTROL PANEL",
                (210,40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.8,
                (255,255,255),
                2)

    # Cards
    cards = [
        ("LIVE FPS", f"{fps:.1f}", (0,255,255)),
        ("PSNR", f"{psnr_value:.2f}", (255,140,0)),
        ("SSIM", f"{ssim_value:.3f}", (0,180,255)),
        ("AVG", f"P:{avg_psnr:.1f}  S:{avg_ssim:.3f}", (220,220,220))
    ]

    x = 35

    for title, val, col in cards:

        rounded_rect(dash, (x,70), (x+290,170), (28,28,28), -1)

        cv2.putText(dash, title, (x+70,105),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (180,180,180), 2)

        cv2.putText(dash, val, (x+55,145),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    col, 2)

        x += 330

    # Graphs
    draw_graph(dash, psnr_history, "LIVE PSNR TREND",
               (255,140,0), 35, 220, 620, 220)

    draw_graph(dash, ssim_history, "LIVE SSIM TREND",
               (0,180,255), 705, 220, 620, 220)

    # Footer
    rounded_rect(dash, (35,470), (1325,535), (22,22,22), -1)

    cv2.putText(dash,
                "SYSTEM ACTIVE  |  RGB + HSV Fusion Running  |  Press Q to Exit",
                (165,512),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,120),
                2)

    # Final output
    final = np.vstack((top, dash))

    cv2.imshow("AI Underwater Enhancement Dashboard", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- END ----------------
cap.release()
cv2.destroyAllWindows()