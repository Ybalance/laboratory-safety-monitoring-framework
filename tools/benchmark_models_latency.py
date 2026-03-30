import time
import cv2
import numpy as np
from ultralytics import YOLO
import os

def benchmark_model(model_path, img_size=640, num_runs=50):
    """
    Benchmarks the model speed by running multiple iterations and averaging the internal speed metrics.
    
    Args:
        model_path (str): Path to the YOLO model file (.pt).
        img_size (int): Size of the input image (assumes square).
        num_runs (int): Number of iterations to average over (after warmup).
    """
    print(f"\nBenchmarking model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        # Load model
        model = YOLO(model_path)
        
        # Check if CUDA is available and force usage
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Create a dummy image (simulating a 640x640 input)
        dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        
        # Warmup (run once to load onto device and initialize)
        print("Warming up...")
        # Run a few times for warmup to be sure
        for _ in range(5):
            model.predict(dummy_img, device=device, verbose=False)
        
        # Run inference for measurement
        print(f"Running benchmark ({num_runs} runs)...")
        
        total_pre = 0.0
        total_inf = 0.0
        total_post = 0.0
        
        for _ in range(num_runs):
            # Synchronize CUDA before starting timing (if needed by external timer, but here we use internal stats)
            if device == 'cuda':
                torch.cuda.synchronize()
                
            results = model.predict(dummy_img, device=device, verbose=False)
            
            # Synchronize CUDA after inference to ensure timing is accurate
            if device == 'cuda':
                torch.cuda.synchronize()

            # Ultralytics results object contains the speed dictionary in ms
            # speed = {'preprocess': float, 'inference': float, 'postprocess': float}
            speed = results[0].speed
            total_pre += speed['preprocess']
            total_inf += speed['inference']
            total_post += speed['postprocess']
            
        avg_pre = total_pre / num_runs
        avg_inf = total_inf / num_runs
        avg_post = total_post / num_runs
        avg_total = avg_pre + avg_inf + avg_post
        
        print(f"Speed: {avg_pre:.2f}ms preprocess, {avg_inf:.2f}ms inference, {avg_post:.2f}ms postprocess per image")
        print(f"Total time per image: {avg_total:.2f}ms")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Model 1
    model_1 = r"c:\Users\23654\Desktop\lab_safety_web_app\yolo26数据\mixup+p2层\best.pt"
    benchmark_model(model_1)
    
    # Model 2
    model_2 = r"c:\Users\23654\Desktop\lab_safety_web_app\新建文件夹\best.pt"
    benchmark_model(model_2)
