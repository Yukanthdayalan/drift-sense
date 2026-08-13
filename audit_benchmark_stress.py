import os
import sys
import math
import json
import time
import subprocess
import csv
from pathlib import Path

def main():
    eval_dir = os.path.abspath("audit_dataset_stress/eval")
    if not os.path.exists(eval_dir):
        print(f"Error: Directory {eval_dir} does not exist.")
        sys.exit(1)
        
    inference_script = os.path.abspath("inference.py")
    python_exe = sys.executable
    
    results = []
    
    samples = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])
    
    if not samples:
        print("No samples found.")
        sys.exit(1)
        
    print(f"Evaluating {len(samples)} samples from {eval_dir}...")
    
    for sample in samples:
        sample_dir = os.path.join(eval_dir, sample)
        ref_path = os.path.join(sample_dir, "reference.png")
        search_path = os.path.join(sample_dir, "search.png")
        gt_path = os.path.join(sample_dir, "ground_truth.json")
        
        if not (os.path.exists(ref_path) and os.path.exists(search_path) and os.path.exists(gt_path)):
            print(f"Skipping {sample} (missing files)")
            continue
            
        try:
            from drift_sense.inference import run_inference
            res = run_inference(ref_path, search_path)
            pred_x = res.prediction.x
            pred_y = res.prediction.y
            runtime = res.execution_time_ms
        except Exception as e:
            print(f"Error running inference on {sample}: {e}")
            continue
            
        # Read ground truth ONLY after inference
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
            
        gt_x = gt_data["gt_center_x"]
        gt_y = gt_data["gt_center_y"]
        
        dx = pred_x - gt_x
        dy = pred_y - gt_y
        err = math.sqrt(dx*dx + dy*dy)
        
        results.append({
            "sample_id": sample,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "error_px": err,
            "runtime_ms": runtime
        })
        print(f"{sample}: Error={err:.4f} px, Time={runtime:.1f} ms")

    if not results:
        print("No valid results computed.")
        sys.exit(1)
        
    errors = [r['error_px'] for r in results]
    runtimes = [r['runtime_ms'] for r in results]
    
    errors_sorted = sorted(errors)
    runtimes_sorted = sorted(runtimes)
    
    count = len(errors)
    mean_err = sum(errors) / count
    med_err = errors_sorted[count // 2]
    max_err = errors_sorted[-1]
    
    acc_1px = sum(1 for e in errors if e <= 1.0) / count * 100
    acc_2px = sum(1 for e in errors if e <= 2.0) / count * 100
    acc_5px = sum(1 for e in errors if e <= 5.0) / count * 100
    
    mean_time = sum(runtimes) / count
    med_time = runtimes_sorted[count // 2]
    max_time = runtimes_sorted[-1]
    
    results_sorted = sorted(results, key=lambda x: x['error_px'])
    worst_case = results_sorted[-1]
    
    print("\n--- RESULTS ---")
    print(f"Number of samples: {count}")
    print(f"Mean error: {mean_err:.4f} px")
    print(f"Median error: {med_err:.4f} px")
    print(f"Maximum error: {max_err:.4f} px")
    print(f"<= 1 px accuracy: {acc_1px:.1f}%")
    print(f"<= 2 px accuracy: {acc_2px:.1f}%")
    print(f"<= 5 px accuracy: {acc_5px:.1f}%")
    print(f"Average runtime: {mean_time:.1f} ms")
    print(f"Median runtime: {med_time:.1f} ms")
    print(f"Maximum runtime: {max_time:.1f} ms")
    print(f"Worst sample: {worst_case['sample_id']} (Error: {worst_case['error_px']:.4f} px)")
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/final_submission_50_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "pred_x", "pred_y", "gt_x", "gt_y", "error_px", "runtime_ms"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
