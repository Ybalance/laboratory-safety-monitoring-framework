import os
import cv2
import glob
import json
import requests
import base64
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import datetime

# --- Configuration ---
MODEL_PATH = r'models/lab_safety_detection6/weights/best.pt'
TEST_IMAGES_DIR = r'yolov8/valid/images'
TEST_LABELS_DIR = r'yolov8/valid/labels'

# VLM Configuration (Same as app.py)
VLM_API_KEY = "sk-nyqmdqemjevzpibcbsicmqjiatxsclohuygdjsvbolgmctze"
VLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct" 

# Target Classes for VLM Verification
# 0: Drinking, 1: Eating
TARGET_CLASS_IDS = [0, 1]
CLASS_NAMES = {0: 'Drinking', 1: 'Eating'}

def call_vlm(image_base64, prompt):
    """Calls the VLM API with retry logic."""
    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "stream": False
    }
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(VLM_API_URL, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                # 成功后也等待几秒，避免连续请求触发限流
                time.sleep(2) 
                return response.json()['choices'][0]['message']['content']
            elif response.status_code == 429: # Rate limit
                wait_time = (2 ** attempt) * 2 # 加大退避时间 (2, 4, 8, 16, 32s...)
                print(f"VLM Rate Limit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"VLM Error: {response.status_code}. Retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"VLM Exception: {e}. Retrying...")
            time.sleep(2)
            
    print("VLM Failed after max retries.")
    return None

def get_ground_truth(label_path):
    """Reads label file and returns set of class IDs present."""
    gt_classes = set()
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    gt_classes.add(int(parts[0]))
    return gt_classes

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))
    # Optional: Limit to first 200 images for quick validation testing
    # image_files = image_files[:200] 
    print(f"Found {len(image_files)} images in {TEST_IMAGES_DIR}")

    # CSV Logging Setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"evaluation_results_{timestamp}.csv"
    
    # Open CSV file
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    
    # Write Header
    header = [
        'Image Name', 
        'Ground Truth', 
        'YOLO Class', 
        'YOLO Confidence', 
        'Action (Strategy)', 
        'VLM Response', 
        'Final Result',
        'Model Parameters'
    ]
    csv_writer.writerow(header)
    print(f"Logging results to {csv_filename}")

    # Metrics
    # Format: [TP, FP, FN]
    stats_yolo = {'Drinking': [0, 0, 0], 'Eating': [0, 0, 0]}
    stats_vlm  = {'Drinking': [0, 0, 0], 'Eating': [0, 0, 0]}
    
    # Global counters for summary
    total_images = 0
    
    print("\nStarting Evaluation...")
    print("-" * 100)
    print(f"{'Image':<30} | {'GT':<10} | {'YOLO Conf':<10} | {'Action':<20} | {'Final':<10}")
    print("-" * 100)

    for img_path in image_files:
        basename = os.path.basename(img_path)
        label_file = basename.replace('.jpg', '.txt')
        label_path = os.path.join(TEST_LABELS_DIR, label_file)
        
        # 1. Ground Truth
        gt_classes = get_ground_truth(label_path)
        has_drinking_gt = 0 in gt_classes
        has_eating_gt = 1 in gt_classes
        
        gt_str = []
        if has_drinking_gt: gt_str.append("Drink")
        if has_eating_gt: gt_str.append("Eat")
        gt_display = ",".join(gt_str) if gt_str else "None"

        # 2. Run YOLO
        img = cv2.imread(img_path)
        if img is None: continue
        
        results = model(img, verbose=False)
        
        # Flags for this image (for overall metrics)
        img_yolo_drinking = False
        img_yolo_eating = False
        img_final_drinking = False
        img_final_eating = False
        
        has_detection = False

        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Check if it's a target class
                if cls_id not in TARGET_CLASS_IDS:
                    continue
                
                has_detection = True
                cls_name = CLASS_NAMES[cls_id]
                is_drinking = (cls_id == 0)
                is_eating = (cls_id == 1)

                # --- YOLO Stats Recording ---
                if is_drinking: img_yolo_drinking = True
                if is_eating: img_yolo_eating = True
                
                # --- Hybrid Logic (Range 0.3 - 0.5) ---
                action_str = ""
                vlm_resp_str = "N/A"
                final_decision = False
                
                if conf >= 0.9:
                    # High confidence: Trust YOLO
                    final_decision = True
                    action_str = "High Conf (>=0.9)"
                elif conf >= 0.3: # And conf < 0.5
                    # Medium confidence: Ask VLM
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        _, buffer = cv2.imencode('.jpg', crop)
                        crop_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        action_prompt = "drinking" if is_drinking else "eating"
                        prompt = f"Is the person in this image {action_prompt}? Please answer strictly with YES or NO. If unsure, answer YES."
                        
                        # Call VLM
                        resp = call_vlm(crop_b64, prompt)
                        vlm_resp_str = str(resp).strip() if resp else "Error"
                        
                        if resp and "YES" in resp.upper() and "NO" not in resp.upper():
                            final_decision = True
                            action_str = "VLM Confirm (YES)"
                        else:
                            final_decision = False # VLM rejected
                            action_str = "VLM Reject (NO)"
                    else:
                        final_decision = False # Bad crop
                        action_str = "Bad Crop"
                else:
                    final_decision = False 
                    action_str = "Low Conf (<0.3)"

                # Update Final Decision Flags
                if final_decision:
                    if is_drinking: img_final_drinking = True
                    if is_eating: img_final_eating = True
                
                # Write to CSV row for this detection
                csv_writer.writerow([
                    basename,
                    gt_display,
                    cls_name,
                    f"{conf:.4f}",
                    action_str,
                    vlm_resp_str,
                    "Keep" if final_decision else "Drop",
                    f"VLM={VLM_MODEL}"
                ])
                
                # Console Output
                print(f"{basename[:30]:<30} | {gt_display:<10} | {conf:.2f} ({cls_name}) | {action_str:<20} | {'Keep' if final_decision else 'Drop'}")

        # If no detections for target classes, log a row saying so
        if not has_detection:
             csv_writer.writerow([
                basename,
                gt_display,
                "None",
                "0.00",
                "No Detection",
                "N/A",
                "N/A",
                f"VLM={VLM_MODEL}"
            ])

        # Update Overall Metrics (Per Image Level)
        # YOLO Stats
        if img_yolo_drinking:
            if has_drinking_gt: stats_yolo['Drinking'][0] += 1 # TP
            else: stats_yolo['Drinking'][1] += 1 # FP
        else:
            if has_drinking_gt: stats_yolo['Drinking'][2] += 1 # FN
            
        if img_yolo_eating:
            if has_eating_gt: stats_yolo['Eating'][0] += 1 # TP
            else: stats_yolo['Eating'][1] += 1 # FP
        else:
            if has_eating_gt: stats_yolo['Eating'][2] += 1 # FN

        # Hybrid/VLM Stats
        if img_final_drinking:
            if has_drinking_gt: stats_vlm['Drinking'][0] += 1 # TP
            else: stats_vlm['Drinking'][1] += 1 # FP
        else:
            if has_drinking_gt: stats_vlm['Drinking'][2] += 1 # FN
            
        if img_final_eating:
            if has_eating_gt: stats_vlm['Eating'][0] += 1 # TP
            else: stats_vlm['Eating'][1] += 1 # FP
        else:
            if has_eating_gt: stats_vlm['Eating'][2] += 1 # FN

    csv_file.close()
    print("-" * 100)
    print(f"Evaluation Complete. Detailed CSV saved to: {csv_filename}")
    
    for cls_name in ['Drinking', 'Eating']:
        print(f"\n--- {cls_name} ---")
        
        # YOLO Stats
        tp, fp, fn = stats_yolo[cls_name]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"[YOLO Only] TP: {tp}, FP: {fp}, FN: {fn} | Precision: {precision:.2f}, Recall: {recall:.2f}")
        
        # VLM Stats
        tp_v, fp_v, fn_v = stats_vlm[cls_name]
        precision_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0
        recall_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0
        print(f"[YOLO + VLM] TP: {tp_v}, FP: {fp_v}, FN: {fn_v} | Precision: {precision_v:.2f}, Recall: {recall_v:.2f}")
        
        if fp > fp_v:
            print(f"-> VLM successfully filtered {fp - fp_v} false positives!")

    # --- Plotting Charts ---
    classes = ['Drinking', 'Eating']
    metrics = ['Precision', 'Recall']
    
    # Prepare data
    yolo_precision = []
    yolo_recall = []
    vlm_precision = []
    vlm_recall = []

    for cls in classes:
        # YOLO
        tp, fp, fn = stats_yolo[cls]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        yolo_precision.append(p)
        yolo_recall.append(r)
        
        # VLM
        tp, fp, fn = stats_vlm[cls]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        vlm_precision.append(p)
        vlm_recall.append(r)

    x = np.arange(len(classes))
    width = 0.35

    # 1. Precision Comparison
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, yolo_precision, width, label='YOLO Only')
    rects2 = ax.bar(x + width/2, vlm_precision, width, label='YOLO + VLM')

    ax.set_ylabel('Precision')
    ax.set_title('Precision Comparison by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    plt.savefig('precision_comparison.png')
    print("\nSaved chart: precision_comparison.png")

    # 2. Recall Comparison
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, yolo_recall, width, label='YOLO Only')
    rects2 = ax.bar(x + width/2, vlm_recall, width, label='YOLO + VLM')

    ax.set_ylabel('Recall')
    ax.set_title('Recall Comparison by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    plt.savefig('recall_comparison.png')
    print("Saved chart: recall_comparison.png")
    
    # 3. False Positives Reduction
    fig, ax = plt.subplots()
    yolo_fps = [stats_yolo[c][1] for c in classes]
    vlm_fps = [stats_vlm[c][1] for c in classes]
    
    rects1 = ax.bar(x - width/2, yolo_fps, width, label='YOLO Only')
    rects2 = ax.bar(x + width/2, vlm_fps, width, label='YOLO + VLM')

    ax.set_ylabel('False Positives (Count)')
    ax.set_title('False Positive Reduction')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.savefig('fp_reduction.png')
    print("Saved chart: fp_reduction.png")

if __name__ == "__main__":
    evaluate()