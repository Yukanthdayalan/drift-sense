import os
import sys
import math
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from drift_sense.inference import run_inference


def evaluate_dataset(eval_dir):
    eval_dir = os.path.join(eval_dir, "eval")
    samples = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])
    
    results = []
    
    for sample in samples:
        sample_dir = os.path.join(eval_dir, sample)
        ref_path = os.path.join(sample_dir, "reference.png")
        search_path = os.path.join(sample_dir, "search.png")
        gt_path = os.path.join(sample_dir, "ground_truth.json")
        
        try:
            res = run_inference(ref_path, search_path)
        except Exception as e:
            print(f"Error on {sample}: {e}")
            continue
            
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
            
        gt_x = gt_data["gt_center_x"]
        gt_y = gt_data["gt_center_y"]
        
        dx = res.prediction.x - gt_x
        dy = res.prediction.y - gt_y
        err = math.sqrt(dx*dx + dy*dy)
        
        results.append(err)
        
    count = len(results)
    if count == 0:
        return {}
        
    results_sorted = sorted(results)
    
    mean_err = sum(results) / count
    med_err = results_sorted[count // 2]
    max_err = results_sorted[-1]
    
    acc_1px = sum(1 for e in results if e <= 1.0) / count * 100
    acc_2px = sum(1 for e in results if e <= 2.0) / count * 100
    acc_5px = sum(1 for e in results if e <= 5.0) / count * 100
    
    return {
        "mean_err": mean_err,
        "med_err": med_err,
        "max_err": max_err,
        "acc_1px": acc_1px,
        "acc_2px": acc_2px,
        "acc_5px": acc_5px,
    }

def main():
    baseline_stats = evaluate_dataset("evaluation_dataset_baseline")
    stress_stats = evaluate_dataset("evaluation_dataset_stress")
    
    out = {
        "baseline": baseline_stats,
        "stress": stress_stats
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/hybrid_comparison.json", "w") as f:
        json.dump(out, f, indent=4)
        
    print("Saved hybrid comparison summary:")
    print(json.dumps(out, indent=4))

if __name__ == "__main__":
    main()
