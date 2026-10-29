"""
=============================================================================
Smart Parking Monitoring and Occupancy Detection System
Subject: Foundations of Computer Vision (FCV) Lab Project
Target: 2nd Year B.Tech (Artificial Intelligence & Data Science)
=============================================================================

This module provides the core computer vision algorithms for detecting
whether individual parking spaces are occupied or vacant in real-world
parking lot imagery and video streams.

Algorithm Pipeline:
1. Load calibrated or auto-generated parking slot bounding boxes (x, y, width, height).
2. Convert the parking lot frame to Grayscale (cv2.cvtColor) to remove color complexity.
3. Apply Gaussian Blur (cv2.GaussianBlur) to suppress camera noise and road grain.
4. Apply Adaptive Gaussian Thresholding (cv2.adaptiveThreshold) to generate
   a binary map highlighting edges, vehicle contours, and texture discontinuities.
5. Apply Median Filtering (cv2.medianBlur) and Morphological Dilation (cv2.dilate)
   to bridge small gaps between vehicle contour segments.
6. For each slot ROI, count the non-zero (white) pixels using cv2.countNonZero():
   - Empty pavement (vacant slot) -> Smooth texture -> Low non-zero pixel count.
   - Parked vehicle (occupied slot) -> Sharp edges, roof seams, windshield, reflection -> High non-zero count.
7. Classify as OCCUPIED (Red rectangle) if pixel count > threshold, else VACANT (Green rectangle).
8. Compute occupancy metrics and overlay a live HUD summary banner.
=============================================================================
"""

import os
import json
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON_PATH = os.path.join(BASE_DIR, "parking_data", "parking_spaces.json")
PKLOT_DIR = os.path.join(BASE_DIR, "dataset", "PKLot")


def check_pklot_available(dataset_path=PKLOT_DIR):
    """
    Check whether the PKLot dataset directory is present on disk.
    
    Returns:
        tuple: (is_available: bool, message: str)
    """
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        contents = os.listdir(dataset_path)
        if len(contents) > 0:
            return True, f"PKLot dataset directory detected at '{dataset_path}' ({len(contents)} items)."
    return False, (
        f"PKLot dataset not found at '{dataset_path}'. "
        "Download instructions and links are provided in README.md."
    )


def load_parking_spaces(json_path=DEFAULT_JSON_PATH, lot_key="real_parking_lot_1"):
    """
    Load parking space bounding boxes from a JSON configuration file.
    
    Parameters:
        json_path (str): Path to the parking_spaces.json file.
        lot_key (str): Key identifying the parking lot configuration.
        
    Returns:
        list of dict: List containing space dictionaries with keys
                      ['id', 'x', 'y', 'w', 'h', 'occupied_ground_truth'].
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Parking configuration file not found at: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    if lot_key not in data:
        raise KeyError(f"Lot '{lot_key}' not found in {json_path}. Available: {list(data.keys())}")
        
    return data[lot_key]["spaces"]


def generate_uniform_grid(img_width, img_height, rows=2, cols=10, 
                          margin_left=50, margin_top=80, 
                          slot_w=None, slot_h=None, gap_x=15, gap_y=60):
    """
    Automatically generate an evenly spaced grid of parking spaces for ANY
    uploaded arbitrary parking lot image.
    
    Parameters:
        img_width (int): Width of the uploaded image in pixels.
        img_height (int): Height of the uploaded image in pixels.
        rows (int): Number of parking rows.
        cols (int): Number of slots per row.
        margin_left (int): Left margin in pixels.
        margin_top (int): Top margin in pixels.
        slot_w (int or None): Width of each slot (auto-computed if None).
        slot_h (int or None): Height of each slot (auto-computed if None).
        gap_x (int): Horizontal spacing between adjacent slots.
        gap_y (int): Vertical drive-lane spacing between rows.
        
    Returns:
        list of dict: List of generated space dicts.
    """
    if slot_w is None:
        usable_w = img_width - 2 * margin_left - (cols - 1) * gap_x
        slot_w = max(30, int(usable_w / cols))
        
    if slot_h is None:
        usable_h = img_height - 2 * margin_top - (rows - 1) * gap_y
        slot_h = max(40, int(usable_h / rows))
        
    spaces = []
    slot_id = 1
    for r in range(rows):
        y = margin_top + r * (slot_h + gap_y)
        for c in range(cols):
            x = margin_left + c * (slot_w + gap_x)
            if x + slot_w <= img_width and y + slot_h <= img_height:
                spaces.append({
                    "id": slot_id,
                    "x": int(x),
                    "y": int(y),
                    "w": int(slot_w),
                    "h": int(slot_h),
                    "occupied_ground_truth": None
                })
                slot_id += 1
                
    return spaces


def save_parking_spaces(json_path, lot_key, spaces_list, image_name="custom_lot.jpg", width=1280, height=720):
    """
    Save or update parking space coordinates in the JSON configuration file.
    """
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
            
    data[lot_key] = {
        "image_name": image_name,
        "width": width,
        "height": height,
        "spaces": spaces_list
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def preprocess_frame(frame, blur_ksize=(3, 3), threshold_block_size=25, threshold_c=16):
    """
    Preprocess the input parking lot frame using foundational CV techniques:
    1. Grayscale Conversion (cv2.cvtColor)
    2. Gaussian Smoothing (cv2.GaussianBlur)
    3. Adaptive Gaussian Thresholding (cv2.adaptiveThreshold)
    4. Median Filtering (cv2.medianBlur)
    5. Morphological Dilation (cv2.dilate)
    
    Parameters:
        frame (np.ndarray): BGR image frame.
        blur_ksize (tuple): Kernel size for Gaussian blur.
        threshold_block_size (int): Size of a pixel neighborhood used to calculate threshold.
        threshold_c (int): Constant subtracted from the mean or weighted mean.
        
    Returns:
        tuple: (gray_image, processed_binary_image)
    """
    # 1. Convert BGR to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_ksize, 1)
    
    # 3. Adaptive Gaussian Thresholding: handles non-uniform illumination and shadows
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        threshold_block_size,
        threshold_c
    )
    
    # 4. Median blur to eliminate salt-and-pepper noise
    median = cv2.medianBlur(thresh, 5)
    
    # 5. Morphological dilation to bridge small gaps in vehicle edge segments
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.dilate(median, kernel, iterations=1)
    
    return gray, processed


def get_default_threshold(spaces):
    """
    Calculate an appropriate default threshold based on average slot size.
    For standard real-world slots (107x48 ~= 5136 px), 900 is optimal.
    For vertical slots (95x190), 100 with inset margin is optimal.
    """
    if not spaces:
        return 900
    avg_area = sum(s["w"] * s["h"] for s in spaces) / len(spaces)
    if avg_area < 8000:
        return 900
    else:
        return 100


def analyze_parking_lot(frame, spaces, threshold_pixels=None, 
                        blur_ksize=(3, 3), threshold_block_size=25, threshold_c=16):
    """
    Analyze parking lot occupancy for each space defined in `spaces`.
    
    Parameters:
        frame (np.ndarray): Input BGR image or video frame.
        spaces (list): List of space dicts with 'id', 'x', 'y', 'w', 'h'.
        threshold_pixels (int or None): Non-zero pixel count threshold above which slot is occupied.
        blur_ksize (tuple): Gaussian kernel size.
        threshold_block_size (int): Block size for adaptive threshold.
        threshold_c (int): Constant for adaptive threshold.
        
    Returns:
        tuple: (annotated_frame, processed_binary, summary_dict, detailed_results)
    """
    if threshold_pixels is None:
        threshold_pixels = get_default_threshold(spaces)
        
    gray, processed = preprocess_frame(
        frame, 
        blur_ksize=blur_ksize, 
        threshold_block_size=threshold_block_size, 
        threshold_c=threshold_c
    )
    
    annotated = frame.copy()
    h_img, w_img = frame.shape[:2]
    
    occupied_count = 0
    vacant_count = 0
    detailed_results = []
    
    # Check if we should use an inner margin for large synthetic bays
    avg_area = (sum(s["w"] * s["h"] for s in spaces) / len(spaces)) if spaces else 5000
    use_inset = (avg_area >= 8000)
    
    for space in spaces:
        slot_id = space["id"]
        x, y, w, h = space["x"], space["y"], space["w"], space["h"]
        
        # Ensure bounding coordinates are within image boundaries
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        
        if use_inset:
            mx = max(2, int(w * 0.10))
            my = max(2, int(h * 0.08))
            inner_x1 = min(x2 - 1, x1 + mx)
            inner_y1 = min(y2 - 1, y1 + my)
            inner_x2 = max(inner_x1 + 1, x2 - mx)
            inner_y2 = max(inner_y1 + 1, y2 - my)
            crop = processed[inner_y1:inner_y2, inner_x1:inner_x2]
        else:
            crop = processed[y1:y2, x1:x2]
            
        # Count non-zero (white) pixels within the cropped slot ROI
        count = int(cv2.countNonZero(crop)) if crop.size > 0 else 0
        
        # Decision rule: Occupied if pixel count > threshold
        is_occupied = count > threshold_pixels
        
        if is_occupied:
            occupied_count += 1
            # Red color for Occupied (BGR: 0, 0, 220)
            color = (0, 0, 220)
            status_text = f"#{slot_id} OCC"
        else:
            vacant_count += 1
            # Green color for Vacant (BGR: 0, 220, 0)
            color = (0, 220, 0)
            status_text = f"#{slot_id} VAC"
            
        detailed_results.append({
            "id": slot_id,
            "x": x, "y": y, "w": w, "h": h,
            "non_zero_pixels": count,
            "is_occupied": bool(is_occupied),
            "ground_truth": space.get("occupied_ground_truth", None)
        })
        
        # Draw bounding rectangle around parking slot (Green for vacant, Red for occupied)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Overlay count text
        label = f"{count}"
        (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated, (x1, y1), (x1 + txt_w + 4, y1 + txt_h + 4), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 + txt_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
        )
        
    total_spaces = len(spaces)
    occupancy_pct = (occupied_count / total_spaces * 100) if total_spaces > 0 else 0.0
    
    # Determine overall lot status dynamically from predictions
    if total_spaces == 0:
        overall_status = "No Spaces Defined"
    elif vacant_count == 0:
        overall_status = "Parking Lot Full"
    elif occupancy_pct >= 85.0:
        overall_status = "Nearly Full"
    else:
        overall_status = "Spaces Available"
        
    summary = {
        "total_spaces": total_spaces,
        "occupied_spaces": occupied_count,
        "vacant_spaces": vacant_count,
        "occupancy_percentage": round(occupancy_pct, 1),
        "status": overall_status,
        "threshold_used": threshold_pixels
    }
    
    # Overlay top HUD summary banner
    draw_summary_hud(annotated, summary)
    
    return annotated, processed, summary, detailed_results


def draw_summary_hud(image, summary):
    """
    Draw a clean HUD banner at the top of the annotated parking lot frame.
    """
    h_img, w_img = image.shape[:2]
    banner_height = 50
    
    # Semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, banner_height), (25, 25, 30), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    
    # Dividing line
    cv2.line(image, (0, banner_height), (w_img, banner_height), (80, 80, 90), 2)
    
    # Status color
    if summary["status"] == "Parking Lot Full":
        status_color = (0, 0, 255) # Red
    elif summary["status"] == "Nearly Full":
        status_color = (0, 165, 255) # Orange
    else:
        status_color = (50, 220, 50) # Green
        
    hud_text = (
        f"TOTAL: {summary['total_spaces']}  |  "
        f"OCCUPIED: {summary['occupied_spaces']} (Red)  |  "
        f"VACANT: {summary['vacant_spaces']} (Green)  |  "
        f"OCCUPANCY: {summary['occupancy_percentage']}%"
    )
    
    cv2.putText(
        image, hud_text, (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2, cv2.LINE_AA
    )
    
    status_badge = f"STATUS: {summary['status'].upper()}"
    (b_w, _), _ = cv2.getTextSize(status_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    cv2.putText(
        image, status_badge, (w_img - b_w - 20, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2, cv2.LINE_AA
    )


def evaluate_detection(detailed_results):
    """
    Calculate evaluation metrics (Accuracy, Precision, Recall, Confusion Matrix)
    against ground truth labels.
    
    Parameters:
        detailed_results (list): Output list from analyze_parking_lot.
        
    Returns:
        dict: Evaluation metrics dictionary.
    """
    tp = 0 # Predicted Occupied, Actually Occupied
    fp = 0 # Predicted Occupied, Actually Vacant
    tn = 0 # Predicted Vacant, Actually Vacant
    fn = 0 # Predicted Vacant, Actually Occupied
    unlabeled = 0
    
    for res in detailed_results:
        gt = res.get("ground_truth")
        if gt is None:
            unlabeled += 1
            continue
            
        pred = res["is_occupied"]
        
        if pred is True and gt is True:
            tp += 1
        elif pred is True and gt is False:
            fp += 1
        elif pred is False and gt is False:
            tn += 1
        elif pred is False and gt is True:
            fn += 1
            
    total_evaluated = tp + fp + tn + fn
    if total_evaluated == 0:
        return {
            "evaluated_spaces": 0,
            "unlabeled_spaces": unlabeled,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        }
        
    accuracy = (tp + tn) / total_evaluated
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "evaluated_spaces": total_evaluated,
        "unlabeled_spaces": unlabeled,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": {
            "TP (True Occupied)": tp,
            "FP (False Occupied)": fp,
            "TN (True Vacant)": tn,
            "FN (False Vacant)": fn
        }
    }


if __name__ == "__main__":
    print("[*] Running parking_detection.py verification on real-world images...")
    
    # 1. Check PKLot dataset presence
    pklot_present, pklot_msg = check_pklot_available()
    print(f"[*] PKLot Dataset Status: {pklot_msg}")
    
    # 2. Test Real-World Parking Lot 1 & 2
    test_lots = [
        ("real_parking_lot_1", "real_parking_lot_1.jpg", 900),
        ("real_parking_lot_2", "real_parking_lot_2.jpg", 900),
        ("sample_lot_1", "parking_lot_1.jpg", 100),
        ("sample_lot_2", "parking_lot_2.jpg", 100)
    ]
    
    for lot_key, img_file, thresh in test_lots:
        img_path = os.path.join(BASE_DIR, "dataset", "images", img_file)
        if not os.path.exists(img_path):
            print(f"[!] File not found: {img_path}")
            continue
            
        spaces = load_parking_spaces(DEFAULT_JSON_PATH, lot_key)
        frame = cv2.imread(img_path)
        
        annotated, processed, summary, details = analyze_parking_lot(
            frame, spaces, threshold_pixels=thresh
        )
        metrics = evaluate_detection(details)
        
        print(f"\n=== Verification Results for {lot_key} ({img_file}) ===")
        print(f"  Total Spaces        : {summary['total_spaces']}")
        print(f"  Predicted Occupied  : {summary['occupied_spaces']}")
        print(f"  Predicted Vacant    : {summary['vacant_spaces']}")
        print(f"  Lot Occupancy Rate  : {summary['occupancy_percentage']}%")
        print(f"  Status              : {summary['status']}")
        print(f"  Evaluation Accuracy : {metrics['accuracy']}%")
        print(f"  Confusion Matrix    : {metrics['confusion_matrix']}")
        
    print("\n[+] Verification complete. Real-world images evaluated successfully.")
