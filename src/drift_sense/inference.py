import argparse
import json
import logging
import sys
import os
from typing import Optional, List, Dict, Any

from drift_sense.matcher import match
from drift_sense.config import EngineConfig
from drift_sense.types import InferenceResult
from drift_sense.exceptions import DriftSenseError


logger = logging.getLogger("drift_sense.inference")


def run_inference(ref_path: str, search_path: str, config: Optional[EngineConfig] = None) -> InferenceResult:
    """
    Run inference on a single pair of reference and search images.

    Args:
        ref_path: Path to the reference image.
        search_path: Path to the search image.
        config: Optional EngineConfig instance. If None, default is used.

    Returns:
        InferenceResult object with the localization results.
    """
    if config is None:
        config = EngineConfig()
    
    return match(ref_path, search_path, config)


def run_batch_inference(image_pairs: List[tuple[str, str]], config: Optional[EngineConfig] = None) -> List[InferenceResult]:
    """
    Run inference on a batch of reference and search images.

    Args:
        image_pairs: A list of tuples (ref_path, search_path).
        config: Optional EngineConfig instance.

    Returns:
        A list of InferenceResult objects.
    """
    results = []
    for ref_path, search_path in image_pairs:
        try:
            res = run_inference(ref_path, search_path, config)
            results.append(res)
        except Exception as e:
            logger.error(f"Failed to process pair ({ref_path}, {search_path}): {e}")
            # Append a dummy failure result or re-raise depending on batch requirements.
            # We'll just raise here to fail fast, or we could return an invalid InferenceResult.
            raise
    return results


def main(args: Optional[List[str]] = None) -> int:
    """
    CLI Entry point for drift_sense inference.
    """
    parser = argparse.ArgumentParser(description="Drift-Sense Inference Engine CLI")
    parser.add_argument("ref", type=str, help="Path to the reference image.")
    parser.add_argument("search", type=str, help="Path to the search image.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to write the JSON result.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    
    parsed = parser.parse_args(args)
    
    if parsed.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s | %(levelname)-8s | %(message)s', stream=sys.stderr)
        
    try:
        result = run_inference(parsed.ref, parsed.search)
        
        output_dict = {
            "prediction_x": result.prediction.x,
            "prediction_y": result.prediction.y,
            "scale_used": result.scale_used,
            "confidence": result.confidence,
            "is_fallback_triggered": result.is_fallback_triggered,
            "execution_time_ms": result.execution_time_ms,
            "candidates_found": result.candidates_found,
            "message": result.message
        }
        
        json_str = json.dumps(output_dict, indent=4)
        
        if parsed.output:
            with open(parsed.output, 'w') as f:
                f.write(json_str)
            logger.info(f"Result written to {parsed.output}")
        else:
            print(json_str)
            
        return 0
        
    except DriftSenseError as e:
        logger.error(f"Inference error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during inference: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
