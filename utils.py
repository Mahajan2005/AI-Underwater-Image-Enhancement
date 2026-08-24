import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return img

def rgb_hsv_fusion(img):
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv / 255.0
    fused = np.concatenate((img, hsv), axis=2)
    return fused

def calculate_metrics(original, enhanced):
    psnr = peak_signal_noise_ratio(original, enhanced, data_range=1.0)
    ssim = structural_similarity(original, enhanced, channel_axis=2, data_range=1.0)
    return psnr, ssim