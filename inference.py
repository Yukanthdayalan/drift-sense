import sys
import os
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from drift_sense.matcher import match
from drift_sense.exceptions import DriftSenseError

import argparse

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Inference")
    parser.add_argument("reference_image_path", type=str, help="Path to reference image")
    parser.add_argument("search_image_path", type=str, help="Path to search image")
    parser.add_argument("--output", type=str, help="Path to write output JSON")
    
    args = parser.parse_args()
    
    ref_path = args.reference_image_path
    search_path = args.search_image_path
    output_path = args.output
    
    if not os.path.exists(ref_path):
        print(f"Error: Reference image not found at '{ref_path}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(search_path):
        print(
            f"Error: Search image not found at '{search_path}'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Keep stdout clean for the mandatory Phase-2 evaluator.
        logging.getLogger("drift_sense").setLevel(logging.CRITICAL)

        result = match(ref_path, search_path)

        x = float(result.prediction.x)
        y = float(result.prediction.y)

        # Optional test/development output.
        if output_path is not None:
            data = {
                "prediction_x": x,
                "prediction_y": y,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        # Mandatory evaluator output.
        print(f"({x:.4f}, {y:.4f})")

    except DriftSenseError as e:
        print(f"Inference failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
