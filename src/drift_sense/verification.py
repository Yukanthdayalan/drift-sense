"""
Verification module for the Drift-Sense inference engine.

Responsible exclusively for verifying the structural similarity of the predicted
sub-pixel coordinate against the reference template. It does not perform any
search or coordinate modification.

Verification Strategy:
    Primary:   Intensity-based ZNCC between the scaled reference and search crop.
    Secondary: Gradient magnitude ZNCC (Sobel) for structural validation.
    Combined:  Weighted combination with configurable weights.

    The intensity ZNCC is the dominant signal because it is robust to the
    independent noise and interpolation artifacts that degrade gradient
    correlation in synthetic FinFET data.
"""
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from drift_sense.config import VerificationConfig
from drift_sense.types import ImageArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    """
    Result of the structural verification stage.
    
    Attributes:
        confidence: Combined verification score in [-1, 1].
        intensity_score: Intensity-based ZNCC score.
        gradient_score: Gradient-based ZNCC score.
        passed: True if the combined confidence >= threshold.
        valid: True if verification successfully ran (e.g., crop was within bounds).
    """
    confidence: float
    intensity_score: float
    gradient_score: float
    passed: bool
    valid: bool


def _compute_gradient_magnitude(img: ImageArray) -> np.ndarray:
    """
    Computes the Sobel gradient magnitude of a 2D float32 image.
    
    Args:
        img: 2D float32 array.
        
    Returns:
        2D float32 array representing the gradient magnitude.
    """
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    return mag


def _scalar_zncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes a scalar Zero Mean Normalized Cross-Correlation (ZNCC)
    between two exactly same-sized 2D arrays.
    
    Args:
        img1: First 2D array.
        img2: Second 2D array.
        
    Returns:
        float score in [-1.0, 1.0]. Returns 0.0 if numerical instability occurs
        (e.g., one of the images has zero variance).
    """
    a = img1.flatten().astype(np.float64)
    b = img2.flatten().astype(np.float64)
    
    a_zm = a - a.mean()
    b_zm = b - b.mean()
    
    var_a = np.sum(a_zm ** 2)
    var_b = np.sum(b_zm ** 2)
    
    if var_a < 1e-8 or var_b < 1e-8:
        return 0.0
        
    numerator = np.sum(a_zm * b_zm)
    denominator = math.sqrt(var_a * var_b)
    
    val = numerator / denominator
    return float(np.clip(val, -1.0, 1.0))


def verify_match(
    search_image: ImageArray,
    reference_image: ImageArray,
    x_sub: float,
    y_sub: float,
    config: VerificationConfig
) -> VerificationResult:
    """
    Extracts a sub-pixel crop from the search image and compares it against
    the reference using a combined intensity + gradient verification score.
    
    The combined score is:
        combined = w_intensity * intensity_zncc + w_gradient * gradient_zncc
    
    where w_intensity and w_gradient are configurable weights that sum to 1.0.
    
    Args:
        search_image: The full 2D search image.
        reference_image: The reference template image (already scaled).
        x_sub: Sub-pixel X coordinate of the top-left corner of the match.
        y_sub: Sub-pixel Y coordinate of the top-left corner of the match.
        config: Verification configuration including the similarity threshold.
        
    Returns:
        VerificationResult object containing confidence and status flags.
    """
    # Validate inputs numerically
    if search_image.size == 0 or reference_image.size == 0:
        return VerificationResult(0.0, 0.0, 0.0, False, False)
        
    if not (np.isfinite(search_image).all() and np.isfinite(reference_image).all()):
        return VerificationResult(0.0, 0.0, 0.0, False, False)
        
    if not (math.isfinite(x_sub) and math.isfinite(y_sub)):
        return VerificationResult(0.0, 0.0, 0.0, False, False)
        
    search_h, search_w = search_image.shape
    ref_h, ref_w = reference_image.shape
    
    if ref_h > search_h or ref_w > search_w:
        return VerificationResult(0.0, 0.0, 0.0, False, False)
        
    # Check boundaries for bilinear interpolation.
    min_x = math.floor(x_sub)
    min_y = math.floor(y_sub)
    max_x = math.ceil(x_sub + ref_w - 1)
    max_y = math.ceil(y_sub + ref_h - 1)
    
    if min_x < 0 or min_y < 0 or max_x >= search_w or max_y >= search_h:
        logger.warning("verify_match: Predicted crop exceeds search boundaries. Returning invalid.")
        return VerificationResult(0.0, 0.0, 0.0, False, False)
        
    # cv2.getRectSubPix expects the center coordinate of the extracted patch.
    x_center = x_sub + (ref_w - 1) / 2.0
    y_center = y_sub + (ref_h - 1) / 2.0
    
    crop = cv2.getRectSubPix(
        search_image.astype(np.float32), 
        (ref_w, ref_h), 
        (x_center, y_center)
    )
    
    if crop is None or crop.shape != (ref_h, ref_w):
        return VerificationResult(0.0, 0.0, 0.0, False, False)
    
    ref_f32 = reference_image.astype(np.float32)
    
    # Primary: Intensity-based ZNCC
    intensity_score = _scalar_zncc(crop, ref_f32)
    
    # Secondary: Gradient-based ZNCC
    g_search = _compute_gradient_magnitude(crop)
    g_ref = _compute_gradient_magnitude(ref_f32)
    gradient_score = _scalar_zncc(g_search, g_ref)
    
    # Combined score with configurable weights
    w_intensity = config.intensity_weight
    w_gradient = config.gradient_weight
    combined = w_intensity * intensity_score + w_gradient * gradient_score
    
    # Clamp combined score to [-1, 1]
    combined = max(-1.0, min(1.0, combined))
    
    passed = bool(combined >= config.min_combined_similarity)
    
    logger.debug(
        "verify_match: intensity=%.4f gradient=%.4f combined=%.4f "
        "(threshold=%.4f) passed=%s",
        intensity_score, gradient_score, combined,
        config.min_combined_similarity, passed
    )
    
    return VerificationResult(
        confidence=combined,
        intensity_score=intensity_score,
        gradient_score=gradient_score,
        passed=passed,
        valid=True
    )
