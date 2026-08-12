"""
Validation utilities for the Drift-Sense inference engine.
Ensures that all inputs meet the strict dimensional and type requirements of the pipeline.
"""
import os
import numpy as np
from drift_sense.types import ImageArray
from drift_sense.exceptions import ImageLoadError, PreprocessingError

def validate_image_path(file_path: str) -> None:
    """
    Validates that a given file path exists and is a file.
    
    Args:
        file_path: Absolute or relative path to the image file.
        
    Raises:
        ImageLoadError: If the file does not exist or is not a valid file.
    """
    if not isinstance(file_path, str):
        raise ImageLoadError(f"File path must be a string, got {type(file_path).__name__}.")
        
    if not os.path.exists(file_path):
        raise ImageLoadError(f"Image file not found: {file_path}")
        
    if not os.path.isfile(file_path):
        raise ImageLoadError(f"Path exists but is not a file: {file_path}")

def validate_image_array(image: ImageArray, min_dim: int = 10) -> None:
    """
    Validates that a loaded image array meets structural requirements.
    
    Args:
        image: The numpy array representing the image.
        min_dim: The minimum acceptable width or height in pixels.
        
    Raises:
        PreprocessingError: If the array is empty, not 2D/3D, or critically undersized.
    """
    if not isinstance(image, np.ndarray):
        raise PreprocessingError(f"Expected numpy array, got {type(image).__name__}.")
        
    if image.size == 0:
        raise PreprocessingError("Image array is empty (size 0).")
        
    if image.ndim not in (2, 3):
        raise PreprocessingError(f"Image array must be 2D (grayscale) or 3D (color), got {image.ndim}D.")
        
    height, width = image.shape[:2]
    if height < min_dim or width < min_dim:
        raise PreprocessingError(f"Image dimensions ({width}x{height}) are smaller than minimum allowed ({min_dim}).")

def validate_reference_search_pairing(ref_image: ImageArray, search_image: ImageArray) -> None:
    """
    Validates that the reference image is strictly smaller than the search image.
    This guarantees the mathematical validity of the correlation boundary conditions.
    
    Args:
        ref_image: The reference template array.
        search_image: The search region array.
        
    Raises:
        PreprocessingError: If the reference is larger than or equal to the search image in any dimension.
    """
    validate_image_array(ref_image)
    validate_image_array(search_image)
    
    ref_h, ref_w = ref_image.shape[:2]
    search_h, search_w = search_image.shape[:2]
    
    if ref_w >= search_w or ref_h >= search_h:
        raise PreprocessingError(
            f"Reference image ({ref_w}x{ref_h}) must be strictly smaller than "
            f"Search image ({search_w}x{search_h}) in all dimensions."
        )
