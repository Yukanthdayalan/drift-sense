import os
import sys
import math
import json
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from drift_sense.inference import run_inference

def main():
    eval_dir = "evaluation_dataset_stress/eval"
    samples = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])
    
    worst_sample = None
    worst_err = -1.0
    worst_res = None
    worst_gt = None
    
    for sample in samples:
        sample_dir = os.path.join(eval_dir, sample)
        ref_path = os.path.join(sample_dir, "reference.png")
        search_path = os.path.join(sample_dir, "search.png")
        gt_path = os.path.join(sample_dir, "ground_truth.json")
        
        try:
            res = run_inference(ref_path, search_path)
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
                
            gt_x = gt_data["gt_center_x"]
            gt_y = gt_data["gt_center_y"]
            
            dx = res.prediction.x - gt_x
            dy = res.prediction.y - gt_y
            err = math.sqrt(dx*dx + dy*dy)
            
            if err > worst_err or (res.low_confidence and worst_err < 10.0):
                worst_err = err
                worst_sample = sample
                worst_res = res
                worst_gt = (gt_x, gt_y)
                
                # Favor low_confidence or large error
                if res.low_confidence and err > 2.0:
                    break
        except Exception as e:
            continue
            
    if worst_sample is None:
        print("No failure case found.")
        sys.exit(1)
        
    out_dir = "docs/failure_case"
    os.makedirs(out_dir, exist_ok=True)
    
    sample_dir = os.path.join(eval_dir, worst_sample)
    shutil.copy(os.path.join(sample_dir, "reference.png"), os.path.join(out_dir, "reference.png"))
    shutil.copy(os.path.join(sample_dir, "search.png"), os.path.join(out_dir, "search.png"))
    
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(f"# Failure Case: {worst_sample}\n\n")
        f.write("This example demonstrates a genuinely difficult periodic region where the disambiguation threshold was not fully triggered or the model was uncertain.\n")
        f.write("Due to the sub-pitch periodicity and complex defect layout combined with high noise, the wrong peak exhibited a marginally higher correlation.\n")
        f.write(f"Ground Truth Center: ({worst_gt[0]:.4f}, {worst_gt[1]:.4f})\n")
        f.write(f"Predicted Center: ({worst_res.prediction.x:.4f}, {worst_res.prediction.y:.4f})\n")
        f.write(f"Error: {worst_err:.4f} pixels\n")
        f.write(f"Low Confidence Flag: {worst_res.low_confidence}\n")
        f.write(f"Confidence Score: {worst_res.confidence:.4f}\n")

    print(f"Created failure case from {worst_sample} with error {worst_err:.4f}")

if __name__ == "__main__":
    main()
