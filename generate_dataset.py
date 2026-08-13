import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from drift_sense.dataset import generate_dataset, GeneratorConfig
except ImportError:
    print("Error: 'drift_sense' package not found.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator for Drift-Sense Evaluator")
    parser.add_argument("--architecture", choices=["finfet", "dram"], default="finfet", help="Semiconductor architecture (default: finfet).")
    parser.add_argument("--num-pairs", type=int, default=10, help="Number of independent sample pairs to generate.")
    parser.add_argument("--output-dir", type=str, default="evaluation_dataset", help="Path to output directory.")
    parser.add_argument("--seed", type=int, default=1000, help="Starting random seed.")
    parser.add_argument("--noise-multiplier", type=float, default=1.0, help="Multiplier for speckle and Gaussian noise parameters.")
    
    args = parser.parse_args()
    
    if args.architecture == "dram":
        print("Warning: DRAM architecture is not yet fully implemented. Falling back to generalized periodic structures similar to FinFET.", file=sys.stderr)
    
    print(f"Generating {args.num_pairs} {args.architecture.upper()} samples in '{args.output_dir}'...")
    
    config = GeneratorConfig(
        noise_sigma_search=8.0 * args.noise_multiplier,
        noise_sigma_ref=5.0 * args.noise_multiplier,
        speckle_sigma=0.05 * args.noise_multiplier,
        num_target_defects=20
    )

    
    try:
        generate_dataset(
            output_directory=args.output_dir,
            number_of_samples=args.num_pairs,
            start_seed=args.seed,
            config=config,
            split="eval"
        )
        print("Dataset generation complete. Ground-truth centers recorded in each sample's ground_truth.json.")
    except Exception as e:
        print(f"Error during dataset generation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
