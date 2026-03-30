import os
import shutil
from ultralytics import YOLO
import cv2

# Configuration
MODEL_PATH = r"c:\Users\23654\Desktop\lab_safety_web_app\yolo26数据\mixup+p2层\best.pt"
INPUT_DIR = r"c:\Users\23654\Desktop\lab_safety_web_app\识别"
OUTPUT_DIR = r"c:\Users\23654\Desktop\lab_safety_web_app\识别_results"

# Target Classes (Drinking=0, Eating=1)
# Adjust these IDs based on your data.yaml if they are different.
# Typically: 0: Drinking, 1: Eating in your dataset based on previous context.
TARGET_CLASSES = [0, 1] 

def run_inference():
    # 1. Setup Output Directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Get Images
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return
        
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Found {len(image_files)} images in {INPUT_DIR}")

    # 3. Run Inference
    print("Running inference...")
    
    for img_file in image_files:
        img_path = os.path.join(INPUT_DIR, img_file)
        
        # Run prediction
        # classes=None means all classes
        results = model.predict(source=img_path, 
                                save=False, 
                                # classes=TARGET_CLASSES, # Commented out to detect ALL classes
                                conf=0.30, # Confidence threshold
                                verbose=False)
        
        # Save Result
        for result in results:
            save_path = os.path.join(OUTPUT_DIR, img_file)
            result.save(filename=save_path) # Built-in save method draws boxes
            print(f"Processed: {img_file}")

    print(f"\nInference complete. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()
