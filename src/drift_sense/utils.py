"""
General utility functions for the Drift-Sense inference engine.
Contains pure math and geometry helpers that do not depend on external pipeline state.
"""
import math
from drift_sense.types import Coordinate

def calculate_distance(p1: Coordinate, p2: Coordinate) -> float:
    """
    Computes the Euclidean distance between two sub-pixel coordinates.
    
    Args:
        p1: First coordinate.
        p2: Second coordinate.
        
    Returns:
        float: Euclidean distance in pixels.
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clamps a floating-point value to be within the specified inclusive range.
    
    Args:
        value: The number to clamp.
        min_val: The lower bound.
        max_val: The upper bound.
        
    Returns:
        float: The clamped value.
        
    Raises:
        ValueError: If min_val > max_val.
    """
    if min_val > max_val:
        raise ValueError(f"min_val ({min_val}) cannot be greater than max_val ({max_val}).")
    return max(min_val, min(value, max_val))

def is_within_bounds(coord: Coordinate, width: int, height: int, margin: float = 0.0) -> bool:
    """
    Checks if a coordinate is strictly within the image dimensions, taking into account
    an optional safety margin.
    
    Args:
        coord: The coordinate to check.
        width: Image width in pixels.
        height: Image height in pixels.
        margin: Safety margin in pixels (defaults to 0.0).
        
    Returns:
        bool: True if the coordinate is within bounds, False otherwise.
        
    Raises:
        ValueError: If margin is negative.
    """
    if margin < 0.0:
        raise ValueError("Margin cannot be negative.")
        
    return (margin <= coord.x <= width - margin) and \
           (margin <= coord.y <= height - margin)
