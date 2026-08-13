import cv2
from typing import Tuple
from drift_sense.types import ImageArray

def get_scaled_dimensions(ref_h: int, ref_w: int, scale: float) -> Tuple[int, int]:
    """
    Computes the scaled dimensions for the reference image.
    
    The physical relationship specifies that the reference image is a small template
    that maps to a larger footprint in the search image. Therefore, the scale factor
    must be applied as a multiplier.
    
    Args:
        ref_h: Original reference height.
        ref_w: Original reference width.
        scale: The scale factor (e.g. ~10.0).
        
    Returns:
        Tuple of (scaled_height, scaled_width).
    """
    scaled_h = max(1, round(ref_h * scale))
    scaled_w = max(1, round(ref_w * scale))
    return scaled_h, scaled_w

def resize_reference_for_scale(ref_image: ImageArray, scale: float) -> ImageArray:
    """
    Resizes the reference image according to the specified scale factor.
    
    Args:
        ref_image: The original reference image array.
        scale: The scale factor.
        
    Returns:
        The resized reference image.
    """
    ref_h, ref_w = ref_image.shape[:2]
    scaled_h, scaled_w = get_scaled_dimensions(ref_h, ref_w, scale)
    return cv2.resize(ref_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
