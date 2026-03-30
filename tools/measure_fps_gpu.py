import time
import torch
from ultralytics import YOLO
import numpy as np
import cv2

def benchmark_pure_inference(model_path, batch_size=1, img_size=640, iterations=500):
    print(f"\n{'='*50}")
    print(f"Benchmarking Pure GPU Inference (Batch Size: {batch_size})")
    print(f"{'='*50}")
    
    # 1. Load Model
    try:
        # Load model
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check GPU
    if not torch.cuda.is_available():
        print("Error: CUDA not available. This script requires a GPU.")
        return
        
    device = torch.device('cuda:0')
    name = torch.cuda.get_device_name(0)
    print(f"Device: {name}")

    # 2. Prepare Data (On GPU directly)
    # Create a random tensor to simulate an image batch
    # Shape: (Batch_Size, 3, Height, Width)
    # We use random data or zeros, it doesn't affect inference speed, only accuracy (which we don't care about here)
    print(f"Allocating memory for input batch ({batch_size}x3x{img_size}x{img_size})...")
    
    # Random float32 input [0, 1] usually expected by model after preprocessing
    input_tensor = torch.rand((batch_size, 3, img_size, img_size), device=device, dtype=torch.float32)
    
    # Note: Ultralytics models usually expect images usually as uint8 [0-255] if passed as numpy,
    # or float [0-1] if passed as tensor.
    # To be closest to internal behavior, we pass the tensor directly.
    
    # 3. Warmup
    print("Warming up GPU...")
    # Run a few passes to initialize CUDA context and optimize graph
    for _ in range(10):
        model.predict(source=input_tensor, verbose=False, device=0, half=True) # Use half precision if supported (default on GPU)

    # 4. Run Benchmark
    print(f"Running {iterations} iterations...")
    
    # Reset peak memory stats to track usage for this specific batch size
    torch.cuda.reset_peak_memory_stats()
    
    # Synchronize before starting timer
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(iterations):
        # We pass the pre-loaded GPU tensor
        # This skips Disk I/O, Image Decoding, and CPU-to-GPU transfer
        model.predict(source=input_tensor, verbose=False, device=0, half=True)

    # Synchronize after finishing
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Get Max Memory Used
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert to MB

    # 5. Calculate Metrics
    total_time = end_time - start_time
    total_frames = iterations * batch_size # Total images processed
    fps = total_frames / total_time
    avg_latency_per_batch = (total_time / iterations) * 1000 # ms
    avg_latency_per_image = avg_latency_per_batch / batch_size

    print(f"\nResults for Batch Size {batch_size}:")
    print(f"Method:        Pure Inference (Pre-allocated GPU Tensors)")
    print(f"Total Time:    {total_time:.4f} s")
    print(f"Total Frames:  {total_frames} ({iterations} batches x {batch_size})")
    print(f"Throughput:    {fps:.2f} FPS")
    print(f"Batch Latency: {avg_latency_per_batch:.2f} ms")
    print(f"Image Latency: {avg_latency_per_image:.2f} ms")
    print(f"GPU Max Mem:   {max_memory:.2f} MB")
    
    return fps

if __name__ == "__main__":
    model_path = r"c:\Users\23654\Desktop\lab_safety_web_app\yolo26数据\mixup+p2层\best.pt"
    
    print("="*60)
    print("YOLO Pure GPU Throughput Benchmark")
    print("Methodology: Pure Inference (No I/O)")
    print("1. Pre-allocates random tensors on GPU VRAM to bypass PCIe/CPU bottlenecks.")
    print("2. Uses torch.cuda.synchronize() for precise kernel timing.")
    print("3. Measures raw model architecture computational speed.")
    print("="*60)
    
    # 1. Real-time Simulation (Batch=1)
    fps_b1 = benchmark_pure_inference(model_path, batch_size=1, iterations=500)

    # 2. Low Latency (Batch=2)
    fps_b2 = benchmark_pure_inference(model_path, batch_size=2, iterations=250)

    # 3. Small Batch (Batch=4)
    fps_b4 = benchmark_pure_inference(model_path, batch_size=4, iterations=200)

    # 4. Medium Batch (Batch=8)
    fps_b8 = benchmark_pure_inference(model_path, batch_size=8, iterations=100)

    # 5. Max Throughput (Batch=16)
    # Reduced iterations for large batches to save time
    fps_b16 = benchmark_pure_inference(model_path, batch_size=16, iterations=100)
    
    # 6. Heavy Load (Batch=32) - Only if VRAM permits
    try:
        fps_b32 = benchmark_pure_inference(model_path, batch_size=32, iterations=50)
    except Exception as e:
        print(f"\nSkipping Batch 32 (OOM likely): {e}")
        fps_b32 = 0

    print("\n" + "="*60)
    print("Final Summary (Pure Inference Speed):")
    print(f"Batch 1  (Latency):    {fps_b1:.2f} FPS")
    print(f"Batch 2  (Throughput): {fps_b2:.2f} FPS")
    print(f"Batch 4  (Throughput): {fps_b4:.2f} FPS")
    print(f"Batch 8  (Throughput): {fps_b8:.2f} FPS")
    print(f"Batch 16 (Throughput): {fps_b16:.2f} FPS")
    if fps_b32 > 0:
        print(f"Batch 32 (Throughput): {fps_b32:.2f} FPS")
    print("="*60)
    print("Note: This measures the maximum theoretical speed of the GPU.")
    print("Real-world speed will be lower due to camera/disk I/O.")
