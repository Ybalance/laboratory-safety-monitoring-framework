import torch
from ultralytics import YOLO
import numpy as np
import time
from pathlib import Path
import yaml
from tqdm import tqdm
import torch.nn.functional as F
import os
import glob

def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes.
    box1: (N, 4)
    box2: (M, 4)
    Returns: (N, M) IoU matrix
    """
    # Ultralytics boxes are usually [x1, y1, x2, y2]
    
    # Area of box1
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    # Area of box2
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Broadcasting for intersection
    lt = torch.max(box1[:, None, :2], box2[:, :2]) # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:]) # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    return inter / (union + 1e-6)

def analyze_redundancy(model_path, data_yaml, conf_threshold=0.25, iou_threshold=0.5):
    print(f"\nAnalyzing model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load dataset info
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get validation images
    val_path_rel = data_config.get('test') or data_config.get('val')
    base = os.path.dirname(data_yaml)
    
    # Try multiple resolution strategies
    candidates = []
    
    # 1. As explicitly defined in YAML (relative to YAML location)
    if val_path_rel:
        if os.path.isabs(val_path_rel):
            candidates.append(val_path_rel)
        else:
            candidates.append(os.path.normpath(os.path.join(base, val_path_rel)))
            
    # 2. Hardcoded common structures relative to YAML
    candidates.append(os.path.join(base, 'test', 'images'))
    candidates.append(os.path.join(base, 'valid', 'images'))
    candidates.append(os.path.join(base, 'val', 'images'))
    
    val_path = None
    for p in candidates:
        if os.path.exists(p):
            val_path = p
            print(f"Resolved image path: {val_path}")
            break
            
    if not val_path:
        print(f"Error: Could not find dataset images. Checked: {candidates}")
        return None
    
    # Handle direct path or path to images dir
    if os.path.isdir(val_path):
        image_files = glob.glob(os.path.join(val_path, '*.jpg')) + glob.glob(os.path.join(val_path, '*.png'))
    else:
        # It might be a text file with paths
        with open(val_path, 'r') as f:
            image_files = [x.strip() for x in f.readlines()]
            
    print(f"Found {len(image_files)} images for analysis.")
    
    # Limit to 200 images for speed if needed, or run all
    # image_files = image_files[:200]
    
    total_images = 0
    total_duplicates = 0
    total_detections = 0
    total_gt = 0
    
    postprocess_times = []
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    for img_path in tqdm(image_files, desc="Processing"):
        # 1. Get Ground Truth
        # Assuming label file is in 'labels' folder parallel to 'images'
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # class x_center y_center width height
                    parts = list(map(float, line.strip().split()))
                    # Ensure we have at least 5 parts (class + 4 bbox)
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # Convert to x1, y1, x2, y2 (normalized)
                        # YOLO format: class x_center y_center width height (normalized 0-1)
                        x, y, w, h = parts[1:5] # Take only the next 4 values
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        gt_boxes.append([x1, y1, x2, y2, cls_id])
                    else:
                        # Malformed line or empty
                        pass
        
        gt_tensor = torch.tensor(gt_boxes, device=device)
        total_gt += len(gt_boxes)
        
        # 2. Run Inference
        # We want to measure post-process time accurately
        # model.predict returns a list of Results objects
        results = model.predict(img_path, conf=conf_threshold, iou=0.7, verbose=False, device=device)
        result = results[0]
        
        # Record time (ms)
        postprocess_times.append(result.speed['postprocess'])
        
        # Get Predictions
        # result.boxes.xyxyn contains normalized coordinates if we ask for it?
        # No, result.boxes.xyxy is absolute pixels. result.boxes.xyxyn is normalized.
        # We need to match coordinate systems. GT is normalized. Let's use normalized for preds.
        pred_boxes = result.boxes.xyxyn # (N, 4)
        pred_cls = result.boxes.cls     # (N,)
        
        if len(pred_boxes) == 0:
            total_images += 1
            continue
            
        total_detections += len(pred_boxes)
        
        # 3. Count Duplicates
        # A duplicate is defined as: A GT object being covered by > 1 prediction (IoU > threshold)
        if len(gt_tensor) > 0:
            # Calculate IoU matrix (N_pred, M_gt)
            # Both are x1, y1, x2, y2 normalized
            ious = box_iou(pred_boxes, gt_tensor[:, :4])
            
            # For each GT, count how many preds matched it
            # A match is IoU > threshold AND class matches
            
            # Expand dims for class comparison
            pred_cls_exp = pred_cls.unsqueeze(1) # (N, 1)
            gt_cls_exp = gt_tensor[:, 4].unsqueeze(0) # (1, M)
            cls_match = (pred_cls_exp == gt_cls_exp) # (N, M)
            
            # Valid matches
            matches = (ious > iou_threshold) & cls_match # (N, M)
            
            # Sum over predictions (dim 0) -> Count of preds per GT
            preds_per_gt = matches.sum(dim=0) # (M,)
            
            # If a GT has 2 preds, it contributes 1 duplicate. If 3, contributes 2.
            # So duplicates = max(0, count - 1)
            img_duplicates = torch.clamp(preds_per_gt - 1, min=0).sum().item()
            total_duplicates += img_duplicates
            
        total_images += 1

    # Statistics
    avg_postprocess = np.mean(postprocess_times)
    avg_duplicates = total_duplicates / total_images if total_images > 0 else 0
    duplicate_ratio = total_duplicates / total_detections if total_detections > 0 else 0
    
    return {
        'model': Path(model_path).name,
        'avg_duplicates_per_img': avg_duplicates,
        'duplicate_ratio': duplicate_ratio,
        'avg_postprocess_ms': avg_postprocess,
        'total_detections': total_detections,
        'total_gt': total_gt
    }

if __name__ == "__main__":
    # Define paths
    baseline_model = r"c:\Users\23654\Desktop\lab_safety_web_app\models\model_test\yolov8_best.pt"
    # Use the latest trained model (Ours)
    ours_model = r"c:\Users\23654\Desktop\lab_safety_web_app\yolo26数据\mixup+p2层\best.pt"
    data_yaml = r"c:\Users\23654\Desktop\lab_safety_web_app\yolov8\data.yaml"
    
    print("="*60)
    print("Verifying Architectural Improvements (Redundancy & Speed)")
    print("="*60)
    
    results = []
    
    # Run analysis
    if os.path.exists(baseline_model):
        res_base = analyze_redundancy(baseline_model, data_yaml)
        if res_base: results.append(res_base)
    else:
        print(f"Baseline model not found: {baseline_model}")
        
    if os.path.exists(ours_model):
        res_ours = analyze_redundancy(ours_model, data_yaml)
        if res_ours: results.append(res_ours)
    else:
        print(f"Ours model not found: {ours_model}")
        
    # Print Comparison Report
    print("\n" + "="*80)
    print(f"{'Metric':<30} | {'Baseline (YOLOv8)':<20} | {'Ours (YOLO26-P2)':<20} | {'Improvement':<10}")
    print("-" * 80)
    
    if len(results) == 2:
        base, ours = results[0], results[1]
        
        # 1. Redundancy (Avg Duplicates)
        dup_imp = ((base['avg_duplicates_per_img'] - ours['avg_duplicates_per_img']) / base['avg_duplicates_per_img'] * 100) if base['avg_duplicates_per_img'] > 0 else 0
        print(f"{'Avg Duplicates / Image':<30} | {base['avg_duplicates_per_img']:.4f}{' '*14} | {ours['avg_duplicates_per_img']:.4f}{' '*14} | -{dup_imp:.1f}%")
        
        # 2. Duplicate Ratio
        ratio_imp = ((base['duplicate_ratio'] - ours['duplicate_ratio']) / base['duplicate_ratio'] * 100) if base['duplicate_ratio'] > 0 else 0
        print(f"{'Duplicate Box Ratio':<30} | {base['duplicate_ratio']:.2%}{' '*14} | {ours['duplicate_ratio']:.2%}{' '*14} | -{ratio_imp:.1f}%")
        
        # 3. Post-process Time
        time_imp = ((base['avg_postprocess_ms'] - ours['avg_postprocess_ms']) / base['avg_postprocess_ms'] * 100) if base['avg_postprocess_ms'] > 0 else 0
        print(f"{'Post-process Time (ms)':<30} | {base['avg_postprocess_ms']:.2f} ms{' '*12} | {ours['avg_postprocess_ms']:.2f} ms{' '*12} | -{time_imp:.1f}%")
        
        print("-" * 80)
        print("\nExperimental Conclusion:")
        if ours['avg_duplicates_per_img'] < base['avg_duplicates_per_img']:
            print(f"- Redundancy Reduced: The proposed model successfully reduced duplicate detections by {dup_imp:.1f}%.")
        if ours['avg_postprocess_ms'] < base['avg_postprocess_ms']:
            print(f"- Inference Speedup: NMS-free/Optimized strategy reduced post-processing latency by {time_imp:.1f}%.")
            
    else:
        print("Comparison requires both models to be available.")
        for res in results:
            print(res)
