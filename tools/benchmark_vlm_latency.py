import os
import time
import base64
import requests
import cv2
import numpy as np

# --- Configuration ---
# Taken from evaluate_vlm.py
VLM_API_KEY = "sk-nyqmdqemjevzpibcbsicmqjiatxsclohuygdjsvbolgmctze"
VLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct" 

# Test Image Path (Using one from the validation set)
TEST_IMAGE_PATH = r"c:\Users\23654\Desktop\lab_safety_web_app\yolov8\valid\images\C1200_MP4-0094_jpg.rf.5fee1ed4a9d7bac1fa6ce688e3892b52.jpg"

def encode_image(image_path):
    """Encodes an image file to base64 string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def benchmark_vlm_latency(iterations=10):
    print(f"{'='*60}")
    print(f"Benchmarking VLM API Latency")
    print(f"Model: {VLM_MODEL}")
    print(f"URL:   {VLM_API_URL}")
    print(f"{'='*60}\n")

    # 1. Prepare Data
    try:
        print(f"Loading test image: {os.path.basename(TEST_IMAGE_PATH)}...")
        image_b64 = encode_image(TEST_IMAGE_PATH)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    prompt = "Is the person in this image drinking? Please answer strictly with YES or NO."
    
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
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "stream": False
    }

    latencies = []
    success_count = 0

    print(f"Starting {iterations} iterations...\n")
    print(f"{'Iter':<5} | {'Status':<10} | {'Latency (s)':<15} | {'Response'}")
    print("-" * 60)

    for i in range(iterations):
        start_time = time.time()
        
        try:
            # Send Request
            response = requests.post(VLM_API_URL, headers=headers, json=data, timeout=60)
            
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            if response.status_code == 200:
                success_count += 1
                try:
                    content = response.json()['choices'][0]['message']['content']
                    # Truncate content for display if too long
                    display_content = (content[:30] + '...') if len(content) > 30 else content
                    display_content = display_content.replace('\n', ' ')
                except:
                    display_content = "Parse Error"
            else:
                display_content = f"Error {response.status_code}"
            
            print(f"{i+1:<5} | {response.status_code:<10} | {latency:.4f}          | {display_content}")
            
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            print(f"{i+1:<5} | {'Fail':<10} | {latency:.4f}          | {str(e)}")

    # Summary
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print("-" * 60)
        print(f"Summary:")
        print(f"Total Requests: {iterations}")
        print(f"Successful:     {success_count}")
        print(f"Average Latency: {avg_latency:.4f} s")
        print(f"Min Latency:     {min_latency:.4f} s")
        print(f"Max Latency:     {max_latency:.4f} s")
    else:
        print("No requests completed.")

if __name__ == "__main__":
    benchmark_vlm_latency(iterations=100)
