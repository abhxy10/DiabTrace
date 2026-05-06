import cv2
import numpy as np
import random

def is_valid_fingerprint(image_path):
    """
    Validates whether the uploaded image is a fingerprint.
    Returns (True, None) if valid, (False, reason) if not.
    Uses ridge structure, contrast, and texture entropy checks.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, "Image could not be read."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Check 1: Image must not be too small ---
    if h < 50 or w < 50:
        return False, "Image is too small to be a fingerprint."

    # --- Check 2: Contrast check — fingerprints have significant dark/light variation ---
    std_dev = float(np.std(gray))
    if std_dev < 20:
        return False, "Image has very low contrast. Please upload a clear fingerprint scan."

    # --- Check 3: Entropy — fingerprints have high texture entropy ---
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    if entropy < 4.5:
        return False, "Image does not appear to have fingerprint texture patterns."

    # --- Check 4: Ridge frequency check using Gabor filter response ---
    # Fingerprints have strong oriented ridge-valley patterns
    gabor_responses = []
    for theta in [0, 45, 90, 135]:
        kern = cv2.getGaborKernel((15, 15), 4.0, np.deg2rad(theta), 8.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
        gabor_responses.append(np.mean(np.abs(filtered)))
    
    max_response = max(gabor_responses)
    if max_response < 5.0:
        return False, "No ridge patterns detected. Please upload a fingerprint image."

    # --- Check 5: Ridge variance ratio — fingerprints have anisotropic texture ---
    # Check if there are periodic horizontal AND vertical structures
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x = np.var(sobelx)
    var_y = np.var(sobely)
    
    if var_x < 100 and var_y < 100:
        return False, "No edge structure found. Upload a proper fingerprint scan."

    # --- Check 6: Color check — reject clearly colorful/photo images ---
    b, g, r = cv2.split(img)
    channel_std = np.std([np.mean(b), np.mean(g), np.mean(r)])
    if channel_std > 25:
        return False, "This appears to be a color photo, not a fingerprint image."

    return True, None


def preprocess_image(image_path):
    """Preprocess fingerprint image: grayscale, denoise, normalize, binarize"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian denoising
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # CLAHE normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(denoised)
    
    # Adaptive thresholding (binarize)
    binary = cv2.adaptiveThreshold(
        normalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return binary, normalized

def count_ridges(binary_img):
    """Count ridges by scanning horizontal lines across the image"""
    h, w = binary_img.shape
    
    # Sample multiple horizontal lines
    ridge_counts = []
    for pct in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y = int(h * pct)
        row = binary_img[y, :]
        
        # Count transitions (white to black = ridge crossing)
        transitions = 0
        for i in range(1, len(row)):
            if row[i-1] == 0 and row[i] == 255:
                transitions += 1
        ridge_counts.append(transitions)
    
    avg_ridges = int(np.mean(ridge_counts))
    # Clamp to realistic range 10-35
    avg_ridges = max(10, min(35, avg_ridges))
    return avg_ridges

def compute_ridge_density(binary_img):
    """Compute ridge density in central 5x5mm ROI"""
    h, w = binary_img.shape
    
    # Central ROI
    cx, cy = w // 2, h // 2
    roi_size = min(h, w) // 4
    roi = binary_img[cy - roi_size:cy + roi_size, cx - roi_size:cx + roi_size]
    
    # Count ridge pixels
    white_pixels = np.sum(roi == 255)
    total_pixels = roi.size
    density_ratio = white_pixels / total_pixels
    
    # Map to realistic ridge density (5-20 ridges/cm2)
    density = 5 + (density_ratio * 15)
    density = max(5.0, min(20.0, round(density, 2)))
    return density

def classify_pattern_type(binary_img):
    """
    Classify fingerprint pattern as Arch, Loop, or Whorl
    based on image analysis (core detection heuristic)
    """
    h, w = binary_img.shape
    
    # Compute horizontal gradient to detect curvature
    sobelx = cv2.Sobel(binary_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary_img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Orientation field
    angle = np.arctan2(sobely, sobelx)
    
    # Split image into left/right halves  
    left_half = angle[:, :w//2]
    right_half = angle[:, w//2:]
    
    # Compute variance of angles in each half
    left_var = np.var(left_half)
    right_var = np.var(right_half)
    total_var = np.var(angle)
    
    # Central region analysis for core detection
    center_region = binary_img[h//3:2*h//3, w//3:2*w//3]
    center_density = np.sum(center_region == 255) / center_region.size
    
    # Heuristic classification
    if total_var > 0.8:
        # High variance = complex pattern (Whorl)
        return 'whorl'
    elif abs(left_var - right_var) < 0.05 and total_var < 0.4:
        # Symmetric low variance = Arch
        return 'arch'
    else:
        # Asymmetric or moderate = Loop
        return 'loop'

def extract_features(image_path):
    """
    Main function: extract all fingerprint features from image.
    Returns dict with pattern_type, ridge_count, ridge_density
    """
    result = preprocess_image(image_path)
    if result is None:
        # Fallback if image can't be read
        return {
            'pattern_type': 'loop',
            'ridge_count': 20,
            'ridge_density': 12.0,
            'error': 'Could not read image'
        }
    
    binary, normalized = result
    
    pattern_type = classify_pattern_type(binary)
    ridge_count = count_ridges(binary)
    ridge_density = compute_ridge_density(binary)
    
    return {
        'pattern_type': pattern_type,
        'ridge_count': ridge_count,
        'ridge_density': ridge_density,
        'error': None
    }
