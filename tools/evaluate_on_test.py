from ultralytics import YOLO
import sys
import os

def evaluate_model():
    model_path = r"c:\Users\23654\Desktop\lab_safety_web_app\yolo26数据\mixup+p2层\best.pt"
    data_yaml = "test_config.yaml"
    
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Starting evaluation on TEST set...")
    # split='test' tells YOLO to evaluate on the 'test' split defined in data.yaml
    results = model.val(data=data_yaml, split='test', plots=True)
    
    print("\n" + "="*50)
    print("Test Set Evaluation Results")
    print("="*50)
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    
    # Save results to a file for easier reading
    with open("test_evaluation_results.txt", "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Set: {data_yaml}\n\n")
        f.write(f"mAP50:    {results.box.map50:.4f}\n")
        f.write(f"mAP50-95: {results.box.map:.4f}\n")
        f.write(f"Precision: {results.box.mp:.4f}\n")
        f.write(f"Recall:    {results.box.mr:.4f}\n")
        # Write per-class metrics if available
        if hasattr(results.box, 'maps'):
            f.write("\nPer-class mAP50-95:\n")
            for i, map_val in enumerate(results.box.maps):
                class_name = results.names[i]
                f.write(f"{class_name}: {map_val:.4f}\n")

    print(f"\nResults saved to test_evaluation_results.txt")

if __name__ == "__main__":
    evaluate_model()
