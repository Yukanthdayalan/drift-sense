"""
Sub-pixel refinement module for the Drift-Sense inference engine.
Performs 1D parabolic interpolation independently along X and Y to refine the 
integer peak coordinates obtained from the ZNCC response map.
"""
import logging
import math
from dataclasses import dataclass

import numpy as np

from drift_sense.config import SubpixelConfig
from drift_sense.types import ImageArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubpixelResult:
    """
    Result of the sub-pixel refinement stage.
    
    Attributes:
        x: Refined X coordinate (column).
        y: Refined Y coordinate (row).
        dx: The calculated sub-pixel shift along X.
        dy: The calculated sub-pixel shift along Y.
        refined: True if any sub-pixel adjustment was successfully made.
    """
    x: float
    y: float
    dx: float
    dy: float
    refined: bool


def _parabolic_offset(zm1: float, z0: float, zp1: float, max_offset: float) -> float:
    """
    Calculates the 1D parabolic sub-pixel offset.
    
    Formula: Δ = (z_-1 - z_+1) / [2 * (z_-1 - 2*z_0 + z_+1)]
    
    Args:
        zm1: Response value at coordinate - 1.
        z0: Response value at coordinate.
        zp1: Response value at coordinate + 1.
        max_offset: Maximum allowed offset to clamp the result.
        
    Returns:
        The calculated sub-pixel offset clamped to [-max_offset, max_offset].
        Returns 0.0 if interpolation is unstable or inputs are non-finite.
    """
    if not (math.isfinite(zm1) and math.isfinite(z0) and math.isfinite(zp1)):
        return 0.0

    denom = 2.0 * (zm1 - 2.0 * z0 + zp1)
    
    # Use epsilon threshold to prevent division by zero or extreme instability
    if abs(denom) < 1e-8:
        return 0.0

    delta = (zm1 - zp1) / denom
    
    # Clamp offset to strict bounds
    delta = max(-max_offset, min(max_offset, delta))
    
    return float(delta)


def refine_subpixel(
    response_map: ImageArray,
    peak_x: int,
    peak_y: int,
    config: SubpixelConfig
) -> SubpixelResult:
    """
    Refines the integer peak coordinates to sub-pixel accuracy.
    
    Args:
        response_map: The 2D float32/float64 ZNCC response map.
        peak_x: The integer X coordinate of the peak.
        peak_y: The integer Y coordinate of the peak.
        config: Sub-pixel configuration containing clamping bounds.
        
    Returns:
        SubpixelResult containing the refined coordinates and computed offsets.
        If refinement is impossible due to boundaries or numerical instability,
        the original integer coordinates are returned with dx=0, dy=0.
    """
    if not isinstance(response_map, np.ndarray) or response_map.ndim != 2:
        raise ValueError("response_map must be a 2D numpy array.")
        
    h, w = response_map.shape
    
    if not (0 <= peak_x < w and 0 <= peak_y < h):
        raise ValueError("peak_x and peak_y must be within the response map bounds.")

    # Boundary check: Cannot interpolate if peak is on the very edge
    if peak_x == 0 or peak_x == w - 1 or peak_y == 0 or peak_y == h - 1:
        logger.debug("refine_subpixel: Peak on boundary, falling back to integer coords.")
        return SubpixelResult(float(peak_x), float(peak_y), 0.0, 0.0, False)

    # Independent X refinement
    z_x_m1 = float(response_map[peak_y, peak_x - 1])
    z_x_0  = float(response_map[peak_y, peak_x])
    z_x_p1 = float(response_map[peak_y, peak_x + 1])
    dx = _parabolic_offset(z_x_m1, z_x_0, z_x_p1, config.max_offset)

    # Independent Y refinement
    z_y_m1 = float(response_map[peak_y - 1, peak_x])
    z_y_0  = float(response_map[peak_y, peak_x])
    z_y_p1 = float(response_map[peak_y + 1, peak_x])
    dy = _parabolic_offset(z_y_m1, z_y_0, z_y_p1, config.max_offset)

    refined = (dx != 0.0 or dy != 0.0)

    logger.debug(
        "refine_subpixel: Refined (%d, %d) by dx=%.4f, dy=%.4f to (%.4f, %.4f)",
        peak_x, peak_y, dx, dy, peak_x + dx, peak_y + dy
    )
    
    return SubpixelResult(
        x=float(peak_x + dx),
        y=float(peak_y + dy),
        dx=dx,
        dy=dy,
        refined=refined
    )
