"""
Image preprocessing pipeline for the Drift-Sense inference engine.
Handles disk I/O, grayscale conversion, optional filtering, and mathematical normalization.
"""
import cv2
import numpy as np
from typing import Tuple

from drift_sense.types import ImageArray
from drift_sense.config import PreprocessingConfig
from drift_sense.exceptions import ImageLoadError
from drift_sense.validate import validate_image_path, validate_image_array

def load_image(file_path: str) -> ImageArray:
    """
    Loads an image from disk and strictly converts it to a grayscale numpy array.
    
    Args:
        file_path: Absolute or relative path to the image file.
        
    Returns:
        ImageArray: 2D numpy array containing the grayscale image (uint8).
        
    Raises:
        ImageLoadError: If the file does not exist, or cv2 fails to decode the pixel data.
    """
    validate_image_path(file_path)
    
    # Load directly as grayscale. OpenCV returns None if it fails to parse the file format.
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ImageLoadError(f"OpenCV failed to decode image data from file: {file_path}")
        
    validate_image_array(image)
    return image

def apply_clahe(image: ImageArray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> ImageArray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Crucial for mitigating local charging effects and uneven illumination in SEM imagery.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # CLAHE algorithm strictly requires uint8 or uint16 in OpenCV.
    if image.dtype != np.uint8:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = normalized.astype(np.uint8)
        
    return clahe.apply(image)

def apply_gaussian_blur(image: ImageArray, kernel_size: Tuple[int, int] = (3, 3), sigma: float = 0.5) -> ImageArray:
    """
    Applies a Gaussian blur. Effectively acts as a low-pass filter to suppress 
    high-frequency Poisson/shot noise before correlation.
    """
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)

def apply_median_blur(image: ImageArray, kernel_size: int = 3) -> ImageArray:
    """
    Applies Median filtering. Highly effective at removing salt-and-pepper sensor noise
    while preserving edge sharpness for structural matching.
    """
    return cv2.medianBlur(image, kernel_size)

def z_score_normalize(image: ImageArray) -> ImageArray:
    """
    Converts image to float32 and applies rigorous Z-score normalization (zero mean, unit variance).
    This mathematically guarantees immunity to global brightness and contrast scaling.
    """
    img_float = image.astype(np.float32)
    mean, stddev = cv2.meanStdDev(img_float)
    
    mean_val = mean[0][0]
    std_val = stddev[0][0]
    
    epsilon = 1e-8
    
    if std_val < epsilon:
        return np.zeros_like(img_float)
        
    normalized = ((img_float - mean_val) / std_val).astype(np.float32)
    return normalized

def preprocess_image(file_path: str, config: PreprocessingConfig) -> ImageArray:
    """
    Executes the sequential preprocessing pipeline on a target image file.
    
    Args:
        file_path: Path to the target image file.
        config: Immutable preprocessing configuration parameters.
        
    Returns:
        ImageArray: The float32 z-score normalized array, ready for mathematical matching.
    """
    image = load_image(file_path)
    
    # Optional contrast equalization
    if config.apply_clahe:
        image = apply_clahe(image, config.clahe_clip_limit, config.clahe_tile_grid_size)
        
    # Optional low-pass filtering
    if config.apply_gaussian_blur:
        image = apply_gaussian_blur(image, config.gaussian_kernel_size, config.gaussian_sigma)
        
    # Mathematical normalization
    normalized = z_score_normalize(image)
    
    return normalized
