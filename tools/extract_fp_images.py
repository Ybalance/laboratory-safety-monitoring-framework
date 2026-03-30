import pandas as pd
import shutil
import os

# Configuration
CSV_FILE = r"c:\Users\23654\Desktop\lab_safety_web_app\置信度2\置信度0.3-0.7\evaluation_results_20260208_131350.csv"
SOURCE_IMAGE_DIR = r"c:\Users\23654\Desktop\lab_safety_web_app\yolov8\valid\images"
OUTPUT_DIR = r"c:\Users\23654\Desktop\lab_safety_web_app\fp_analysis_0.3_0.7"

def parse_ground_truth(gt_str):
    labels = set()
    if pd.isna(gt_str) or str(gt_str).lower() == "none":
        return labels
    
    parts = str(gt_str).split(',')
    for p in parts:
        p = p.strip()
        if p == "Drink":
            labels.add("Drinking")
        elif p == "Eat":
            labels.add("Eating")
    return labels

def copy_fp_images():
    # 1. Setup Directories
    yolo_fp_dir = os.path.join(OUTPUT_DIR, "YOLO_Only_FP")
    vlm_fp_dir = os.path.join(OUTPUT_DIR, "YOLO_VLM_FP")
    filtered_fp_dir = os.path.join(OUTPUT_DIR, "Filtered_FP_Differences") # New folder
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(yolo_fp_dir)
    os.makedirs(vlm_fp_dir)
    os.makedirs(filtered_fp_dir) # Create new folder

    print(f"Reading CSV: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Group by Image Name to handle multiple detections per image
    grouped = df.groupby('Image Name')
    
    yolo_fp_count = 0
    vlm_fp_count = 0
    diff_count = 0

    print("Processing images...")
    
    for img_name, group in grouped:
        # Get Ground Truth for this image
        gt_str = group.iloc[0]['Ground Truth']
        gt_set = parse_ground_truth(gt_str)
        
        # 1. Determine Predictions for this image (Set level)
        pred_set_yolo = set()
        pred_set_vlm = set()
        
        for _, row in group.iterrows():
            cls = row['YOLO Class']
            if pd.isna(cls) or str(cls).lower() == "none" or cls == "None":
                continue
            
            if cls == 'Drink': cls = 'Drinking'
            if cls == 'Eat': cls = 'Eating'
            if cls not in ['Drinking', 'Eating']:
                continue
            
            # YOLO: All detections in CSV
            pred_set_yolo.add(cls)
            
            # VLM: Only 'Keep'
            if row['Final Result'] == 'Keep':
                pred_set_vlm.add(cls)
        
        # 2. Check FP for each class (Set Logic)
        
        for cls in ['Drinking', 'Eating']:
            is_pos_gt = cls in gt_set
            
            # Check FP status
            is_fp_yolo = (not is_pos_gt) and (cls in pred_set_yolo)
            is_fp_vlm = (not is_pos_gt) and (cls in pred_set_vlm)
            
            # --- YOLO FP ---
            if is_fp_yolo:
                yolo_fp_count += 1
                src_path = os.path.join(SOURCE_IMAGE_DIR, img_name)
                dst_name = f"{yolo_fp_count:03d}_{cls}_{img_name}"
                dst_path = os.path.join(yolo_fp_dir, dst_name)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
            
            # --- VLM FP ---
            if is_fp_vlm:
                vlm_fp_count += 1
                src_path = os.path.join(SOURCE_IMAGE_DIR, img_name)
                dst_name = f"{vlm_fp_count:03d}_{cls}_{img_name}"
                dst_path = os.path.join(vlm_fp_dir, dst_name)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)

            # --- Difference (YOLO has FP but VLM does not) ---
            # This means VLM successfully filtered out the FP
            if is_fp_yolo and not is_fp_vlm:
                diff_count += 1
                src_path = os.path.join(SOURCE_IMAGE_DIR, img_name)
                # Save as: Count_ClassName_ImageName
                dst_name = f"{diff_count:03d}_{cls}_{img_name}"
                dst_path = os.path.join(filtered_fp_dir, dst_name)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)

    print("-" * 60)
    print(f"Extraction Complete.")
    print(f"YOLO Only FP Count: {yolo_fp_count}")
    print(f"YOLO+VLM FP Count:  {vlm_fp_count}")
    print(f"Filtered (Diff) Count: {diff_count} (Expected: {yolo_fp_count - vlm_fp_count})")
    print(f"Images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    copy_fp_images()
