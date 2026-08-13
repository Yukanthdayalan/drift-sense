import cv2
from typing import Tuple
from drift_sense.types import ImageArray

def get_scaled_dimensions(ref_h: int, ref_w: int, scale: float) -> Tuple[int, int]:
    """
    Computes the scaled dimensions for the reference image.
    
    Reference is the high-magnification/native-detail image and is therefore 
    larger than its corresponding Search footprint. To match Search coordinates, 
    the Reference is downscaled by the estimated scale factor.
    
    Args:
        ref_h: Original reference height.
        ref_w: Original reference width.
        scale: The scale factor (e.g. ~10.0).
        
    Returns:
        Tuple of (scaled_height, scaled_width).
    """
    scaled_h = max(1, int(round(ref_h / scale)))
    scaled_w = max(1, int(round(ref_w / scale)))
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
    return cv2.resize(ref_image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
