"""
Custom exception definitions for the Drift-Sense inference engine.
Establishes a strict error hierarchy for deterministic fallback handling.
"""

class DriftSenseError(Exception):
    """
    Base exception class for all Drift-Sense engine errors.
    All custom exceptions must inherit from this to allow top-level catching.
    """
    pass

class ImageLoadError(DriftSenseError):
    """
    Raised when an image cannot be read from disk, is corrupted, or has an invalid format.
    """
    pass

class PreprocessingError(DriftSenseError):
    """
    Raised when an error occurs during image normalization, filtering, or type conversion.
    """
    pass

class ScaleSearchError(DriftSenseError):
    """
    Raised when scale estimation fails, such as PSD ratio falling completely 
    outside expected bounds or Coarse Search yielding no viable correlation peaks.
    """
    pass

class MatchingError(DriftSenseError):
    """
    Raised when ZNCC mathematics become unstable, specifically during scenarios 
    like zero local variance (featureless regions) leading to division by zero.
    """
    pass

class VerificationError(DriftSenseError):
    """
    Raised when the final candidate fails the structural gradient verification stage,
    indicating a catastrophic mismatch despite a mathematically valid peak.
    """
    pass

class PeakDetectionError(DriftSenseError):
    """
    Raised when peak detection fails to find any valid candidate in the ZNCC
    response map, e.g. due to a degenerate map (all NaN/Inf), an empty response,
    or no peak surviving the response-delta threshold.
    """
    pass
