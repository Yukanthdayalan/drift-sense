"""
End-to-end inference orchestrator for the Drift-Sense pipeline.
Connects all deterministic modules into a single execution path.
"""
import logging
import time

import cv2

from drift_sense.config import EngineConfig, get_default_config
from drift_sense.exceptions import DriftSenseError
from drift_sense.types import Coordinate, InferenceResult
from drift_sense.validate import validate_image_path, validate_reference_search_pairing
from drift_sense.preprocess import preprocess_image
from drift_sense.scale_search import estimate_top_n_scales
from drift_sense.zncc import compute_zncc_fft
from drift_sense.peak_detector import detect_best_peak, get_sharpness
from drift_sense.subpixel import refine_subpixel
from drift_sense.verification import verify_match

logger = logging.getLogger(__name__)

TOP_2_INTENSITY_THRESHOLD = 0.015

def match(reference_path: str, search_path: str, config: EngineConfig = None) -> InferenceResult:
    start_time = time.perf_counter()
    if config is None:
        config = get_default_config()

    try:
        validate_image_path(reference_path)
        validate_image_path(search_path)
        
        ref_norm = preprocess_image(reference_path, config.preprocessing)
        search_norm = preprocess_image(search_path, config.preprocessing)
        
        validate_reference_search_pairing(ref_norm, search_norm)
        
        scales = estimate_top_n_scales(ref_norm, search_norm, config.scale_search, n_scales=2)
        
        from drift_sense.geometry import get_scaled_dimensions, resize_reference_for_scale
        cands = []
        for scale_val in scales:
            ref_h, ref_w = ref_norm.shape
            scaled_h, scaled_w = get_scaled_dimensions(ref_h, ref_w, scale_val)
            
            if scaled_h >= search_norm.shape[0] or scaled_w >= search_norm.shape[1]:
                continue
                
            for angle in [-15.0, -7.5, 0.0, 7.5, 15.0]:
                if angle != 0.0:
                    rot_mat = cv2.getRotationMatrix2D((ref_w / 2.0, ref_h / 2.0), angle, 1.0)
                    ref_rot = cv2.warpAffine(ref_norm, rot_mat, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                else:
                    ref_rot = ref_norm
                    
                ref_scaled = resize_reference_for_scale(ref_rot, scale_val)
                response_map = compute_zncc_fft(ref_scaled, search_norm)
                
                best_peak = detect_best_peak(response_map, config.nms, config.tie_break)
                sharpness = get_sharpness(response_map, best_peak.x, best_peak.y)
                sub_result = refine_subpixel(response_map, best_peak.x, best_peak.y, config.subpixel)
                
                x_center = sub_result.x + (scaled_w - 1) / 2.0
                y_center = sub_result.y + (scaled_h - 1) / 2.0
                
                cands.append({
                    "scale": scale_val,
                    "angle": angle,
                    "x_center": x_center,
                    "y_center": y_center,
                    "intensity": float(best_peak.score),
                    "sharpness": sharpness,
                    "ref_scaled": ref_scaled,
                    "sub_result": sub_result,
                })
            
        if not cands:
            raise DriftSenseError("No valid candidate scales found after ZNCC scoring.")
            
        cands.sort(key=lambda c: c["intensity"], reverse=True)
        best_cand = cands[0]
        
        if len(cands) >= 2:
            diff = cands[0]["intensity"] - cands[1]["intensity"]
            if diff < TOP_2_INTENSITY_THRESHOLD:
                disambiguation_method = "sharpness"
                selected_idx = 0
                
                # Check for CNN
                cnn_model = None
                if getattr(config, 'use_cnn_disambiguation', True):
                    try:
                        from drift_sense.disambiguator import load_disambiguator
                        cnn_model = load_disambiguator()
                    except ImportError:
                        pass
                        
                if cnn_model is not None:
                    import torch
                    from drift_sense.disambiguator import extract_crop
                    
                    crop0 = extract_crop(search_norm, cands[0]["x_center"], cands[0]["y_center"], 64)
                    crop1 = extract_crop(search_norm, cands[1]["x_center"], cands[1]["y_center"], 64)
                    
                    t0 = torch.from_numpy(crop0).float().unsqueeze(0).unsqueeze(0)
                    t1 = torch.from_numpy(crop1).float().unsqueeze(0).unsqueeze(0)
                    
                    with torch.no_grad():
                        score0 = cnn_model(t0).item()
                        score1 = cnn_model(t1).item()
                        
                    logger.info(f"CNN scores: cand0={score0:.3f}, cand1={score1:.3f}")
                    
                    sharp_pick = 1 if cands[1]["sharpness"] > cands[0]["sharpness"] else 0
                    
                    if sharp_pick == 0 and score0 > 0.5:
                        selected_idx = 0
                        disambiguation_method = "cnn_override (agreed)"
                    elif sharp_pick == 1 and score1 > 0.5 and score1 > score0 + 0.1:
                        selected_idx = 1
                        disambiguation_method = "cnn_override (agreed)"
                    elif score1 > 0.7 and score0 < 0.3:
                        selected_idx = 1
                        disambiguation_method = "cnn_override"
                    elif score0 > 0.7 and score1 < 0.3:
                        selected_idx = 0
                        disambiguation_method = "cnn_override"
                    else:
                        selected_idx = sharp_pick
                else:
                    selected_idx = 1 if cands[1]["sharpness"] > cands[0]["sharpness"] else 0
                    
                if selected_idx == 1:
                    logger.info(f"Top-2 Disambiguation triggered ({disambiguation_method}): candidate 2 selected.")
                    best_cand = cands[1]
                else:
                    logger.info(f"Top-2 Disambiguation triggered ({disambiguation_method}): candidate 1 selected.")
        else:
            disambiguation_method = "unambiguous"
                    
        verification = verify_match(
            search_norm, 
            best_cand["ref_scaled"], 
            best_cand["sub_result"].x, 
            best_cand["sub_result"].y, 
            config.verification
        )
        
        execution_time = (time.perf_counter() - start_time) * 1000.0
        
        return InferenceResult(
            prediction=Coordinate(x=best_cand["x_center"], y=best_cand["y_center"]),
            scale_used=best_cand["scale"],
            confidence=verification.confidence,
            is_fallback_triggered=not verification.passed,
            execution_time_ms=execution_time,
            low_confidence=verification.confidence < config.verification.low_confidence_threshold,
            message="Success" if verification.passed else "Verification Failed"
        )
        
    except DriftSenseError as e:
        logger.error("Inference failed: %s", str(e))
        raise
