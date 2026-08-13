"""
Configuration module for the Drift-Sense inference engine.
Defines immutable, type-checked configuration structures for all pipeline stages.
"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for image preprocessing stage."""
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    apply_gaussian_blur: bool = False
    gaussian_kernel_size: Tuple[int, int] = (3, 3)
    gaussian_sigma: float = 0.5

@dataclass(frozen=True)
class ScaleSearchConfig:
    """Configuration for scale estimation and search bracket."""
    use_psd_prior: bool = True
    min_scale: float = 9.0
    max_scale: float = 11.0
    coarse_step: float = 0.5
    fine_step: float = 0.1
    ultra_fine_step: float = 0.02
    psd_confidence_threshold: float = 3.0  # Minimum peak prominence to trust PSD

@dataclass(frozen=True)
class ZNCCConfig:
    """Configuration for the FFT-ZNCC matching engine."""
    epsilon: float = 1e-8  # Numerical stability constant for division

@dataclass(frozen=True)
class NMSConfig:
    """Configuration for Adaptive Non-Maximum Suppression."""
    suppression_radius: int = 5
    top_k: int = 10
    score_threshold: float = 0.5
    response_delta: float = 0.05  # Relative threshold: keep scores >= (max - response_delta)

@dataclass(frozen=True)
class TieBreakConfig:
    delta: float = 0.005  # Score equivalence threshold for tie breaking

@dataclass(frozen=True)
class SubpixelConfig:
    """Configuration for sub-pixel parabolic refinement."""
    max_offset: float = 0.5  # Maximum allowed shift before clamping

@dataclass(frozen=True)
class VerificationConfig:
    """Configuration for final crop validation.
    
    Uses a combined intensity + gradient ZNCC score for verification.
    The combined score = intensity_weight * intensity_zncc + gradient_weight * gradient_zncc.
    
    Default weights (0.8 intensity, 0.2 gradient) strongly favor the intensity-based
    signal, which is robust to independent noise and interpolation artifacts that
    degrade gradient correlation in synthetic FinFET data.
    """
    min_combined_similarity: float = 0.20
    # Threshold for flagging low_confidence in output.
    # Chosen to be midway between the fallback threshold (0.2) and a strong match (0.8+)
    # to flag marginally acceptable matches that didn't outright fail.
    low_confidence_threshold: float = 0.50
    intensity_weight: float = 0.8
    gradient_weight: float = 0.2
    # Legacy field preserved for backward compatibility with tests
    min_gradient_similarity: float = 0.4

@dataclass(frozen=True)
class EngineConfig:
    """Root configuration object encompassing all pipeline stages."""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    scale_search: ScaleSearchConfig = field(default_factory=ScaleSearchConfig)
    zncc: ZNCCConfig = field(default_factory=ZNCCConfig)
    nms: NMSConfig = field(default_factory=NMSConfig)
    tie_break: TieBreakConfig = field(default_factory=TieBreakConfig)
    subpixel: SubpixelConfig = field(default_factory=SubpixelConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

def get_default_config() -> EngineConfig:
    """
    Instantiates and returns the default engine configuration.
    
    Returns:
        EngineConfig: The immutable default configuration.
    """
    return EngineConfig()
