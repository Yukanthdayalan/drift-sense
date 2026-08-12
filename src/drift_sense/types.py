"""
Type definitions and data structures for the Drift-Sense inference engine.
Centralizes domain-specific types to ensure strict typing across all modules.
"""
from dataclasses import dataclass
from typing import TypeAlias, Any
import numpy as np

# Type alias for 2D numpy arrays representing images (typically float32 or uint8)
ImageArray: TypeAlias = np.ndarray

@dataclass(frozen=True)
class Coordinate:
    """
    Represents a sub-pixel 2D coordinate in the image space.
    """
    x: float
    y: float

@dataclass(frozen=True)
class MatchCandidate:
    """
    Represents a localized peak found during the ZNCC search phase.
    Mutable attributes are omitted; candidate evolution should yield new instances or use secondary structures.
    """
    x: float
    y: float
    scale: float
    ncc_score: float
    peak_ratio: float = 0.0
    distance_to_center: float = 0.0
    final_score: float = 0.0

@dataclass(frozen=True)
class InferenceResult:
    """
    The final output payload of the inference pipeline, structured for direct CSV serialization 
    in the benchmarking framework.
    """
    prediction: Coordinate
    scale_used: float
    confidence: float
    is_fallback_triggered: bool
    execution_time_ms: float
    candidates_found: int = 0
    message: str = "Success"
