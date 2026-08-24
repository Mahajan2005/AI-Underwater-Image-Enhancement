import torch
import numpy as np
import matplotlib.pyplot as plt

from model import UnderwaterEnhancer
from utils import load_image, rgb_hsv_fusion, calculate_metrics

# Load image
image_path = "sample.jpg"
original_img = load_image(image_path)

# RGB + HSV feature fusion
fused_img = rgb_hsv_fusion(original_img)
fused_img = np.transpose(fused_img, (2, 0, 1))  # HWC -> CHW
fused_img = torch.tensor(fused_img, dtype=torch.float32).unsqueeze(0)

# Load model
model = UnderwaterEnhancer()
model.eval()

# Enhance image
with torch.no_grad():
    output = model(fused_img)

enhanced_img = output.squeeze(0).permute(1, 2, 0).numpy()

# Calculate metrics
psnr, ssim = calculate_metrics(original_img, enhanced_img)

print(f"PSNR: {psnr:.2f}")
print(f"SSIM: {ssim:.4f}")

# Save output instead of showing window
plt.imsave("enhanced_output.jpg", enhanced_img)
print("Enhanced image saved as enhanced_output.jpg")