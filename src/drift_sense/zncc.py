"""
Zero Mean Normalized Cross-Correlation (ZNCC) engine for the Drift-Sense pipeline.

This module is the exclusive mathematical core for computing spatial correlation
between a resized reference template and a search image. It returns ONLY a
response map — coordinate selection belongs to downstream modules.

ARCHITECTURAL CONTRACT:
  - This module does NOT select peaks, perform NMS, tie-break, or subpixel-refine.
  - The FFT implementation is the production path.
  - The spatial implementation is the verification/debugging reference.
  - All outputs are float32 response maps in the valid correlation domain [-1, 1].
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import scipy.signal

from drift_sense.exceptions import MatchingError, PreprocessingError
from drift_sense.logging_utils import get_logger
from drift_sense.types import ImageArray
from drift_sense.validate import validate_image_array

logger: logging.Logger = get_logger(__name__)

# Numerical stability constant — applied universally to every denominator.
_EPSILON: float = 1e-8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_float32(image: ImageArray, name: str) -> None:
    """
    Raises PreprocessingError if the image is not float32.

    Args:
        image: The array to inspect.
        name:  Human-readable label for error messages.

    Raises:
        PreprocessingError: If dtype is not float32.
    """
    if image.dtype != np.float32:
        raise PreprocessingError(
            f"{name} must be float32, got {image.dtype}. "
            "Run preprocess.z_score_normalize() before calling ZNCC."
        )


def _assert_2d(image: ImageArray, name: str) -> None:
    """
    Raises PreprocessingError if the image is not a 2D array.

    Args:
        image: The array to inspect.
        name:  Human-readable label for error messages.

    Raises:
        PreprocessingError: If array is not 2-dimensional.
    """
    if image.ndim != 2:
        raise PreprocessingError(
            f"{name} must be 2D (grayscale), got {image.ndim}D."
        )


def _check_no_nan_inf(image: ImageArray, name: str) -> None:
    """
    Raises MatchingError if the image contains NaN or Inf values.

    Args:
        image: The array to inspect.
        name:  Human-readable label for error messages.

    Raises:
        MatchingError: If any element is NaN or Inf.
    """
    if not np.isfinite(image).all():
        raise MatchingError(
            f"{name} contains NaN or Inf values. "
            "Ensure the preprocessing pipeline produced clean output."
        )


def _validate_zncc_inputs(template: ImageArray, search: ImageArray) -> None:
    """
    Centralised input validation gate shared by both ZNCC implementations.

    Args:
        template: The zero-mean normalised reference template (float32, 2D).
        search:   The zero-mean normalised search image (float32, 2D).

    Raises:
        PreprocessingError: On dtype or dimensionality violations.
        MatchingError:      On NaN/Inf content or dimensional impossibility.
    """
    validate_image_array(template, min_dim=3)
    validate_image_array(search, min_dim=3)
    _assert_float32(template, "template")
    _assert_float32(search, "search")
    _assert_2d(template, "template")
    _assert_2d(search, "search")
    _check_no_nan_inf(template, "template")
    _check_no_nan_inf(search, "search")

    t_h, t_w = template.shape
    s_h, s_w = search.shape
    if t_h >= s_h or t_w >= s_w:
        raise MatchingError(
            f"Template ({t_w}x{t_h}) must be strictly smaller than "
            f"search image ({s_w}x{s_h}) in both dimensions."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_template(template: ImageArray) -> ImageArray:
    """
    Subtracts the template mean and divides by its standard deviation.

    Produces a zero-mean, unit-variance version of the template patch,
    which is the canonical input to ZNCC numerator computation.

    Purpose:
        Ensures ZNCC is immune to global brightness/contrast differences
        between the reference and search images.

    Inputs:
        template: 2D float32 array of any size.

    Outputs:
        2D float32 array with mean ≈ 0 and std ≈ 1.
        If std < epsilon (flat patch), returns a zero array of the same shape.

    Raises:
        PreprocessingError: If template is not 2D float32.

    Complexity: O(N) where N = template pixels.

    Numerical Notes:
        - std is lower-bounded by _EPSILON before division.
        - Result is explicitly cast to float32 to prevent float64 upcasting.

    Industrial Notes:
        Featureless reference pads (std ≈ 0) are handled defensively;
        the zero output propagates through ZNCC as a legitimately zero
        correlation surface, which the downstream peak detector will
        correctly classify as a low-confidence result.
    """
    _assert_2d(template, "template")
    _assert_float32(template, "template")

    mean = float(template.mean())
    std = float(template.std())

    if std < _EPSILON:
        logger.warning("normalize_template: flat template detected (std < epsilon). Returning zeros.")
        return np.zeros_like(template)

    return ((template - mean) / std).astype(np.float32)


def compute_local_statistics(
    search: ImageArray,
    template_h: int,
    template_w: int,
) -> Tuple[ImageArray, ImageArray]:
    """
    Computes the local sliding-window mean and standard deviation of the
    search image for every valid template position using integral images.

    These are the denominator components required by the spatial ZNCC formula.

    Purpose:
        Efficiently generates the per-position normalisation factors for the
        spatial ZNCC implementation without recomputing per window.

    Inputs:
        search:     2D float32 normalised search image.
        template_h: Template height in pixels.
        template_w: Template width in pixels.

    Outputs:
        Tuple (local_mean, local_std):
            - local_mean: float32 array of shape (s_h - t_h + 1, s_w - t_w + 1).
            - local_std:  float32 array of the same shape, lower-bounded by 0.

    Raises:
        PreprocessingError: On invalid inputs.
        MatchingError:      If template dimensions exceed search dimensions.

    Complexity:
        O(N_s) using OpenCV box filter as an O(1)-per-pixel sliding sum.

    Numerical Notes:
        - local_var is clamped to max(local_var, 0) before sqrt to guard against
          floating-point cancellation producing tiny negative values.
        - local_std is lower-bounded by _EPSILON.

    Industrial Notes:
        This function is shared between spatial ZNCC and the FFT normalisation
        denominator. Keeping it separate enforces single-responsibility and
        allows independent unit testing.
    """
    _assert_2d(search, "search")
    _assert_float32(search, "search")

    n_pixels = float(template_h * template_w)
    ones = np.ones((template_h, template_w), dtype=np.float64)
    search_f64 = search.astype(np.float64)

    # Use fftconvolve with a flat ones kernel in 'valid' mode.
    # This produces sums over windows [r:r+t_h, c:c+t_w] — identical coordinate
    # alignment to the ZNCC numerator computed via fftconvolve(search, flip(tmpl), 'valid').
    # cv2.boxFilter is NOT used here because its default centered anchor desynchronises
    # position (r,c) from the top-left-aligned correlation window.
    sum_s = scipy.signal.fftconvolve(search_f64, ones, mode="valid")
    sum_s2 = scipy.signal.fftconvolve(search_f64 ** 2, ones, mode="valid")

    local_mean = (sum_s / n_pixels).astype(np.float32)

    # Clamp to [0, ∞) before sqrt — floating-point cancellation can yield tiny negatives.
    local_var = sum_s2 / n_pixels - (sum_s / n_pixels) ** 2
    local_var = np.maximum(local_var, 0.0)
    local_std = np.sqrt(local_var).astype(np.float32)

    return local_mean, local_std


def validate_response(response_map: ImageArray) -> Tuple[float, float]:
    """
    Validates a ZNCC response map and returns its scalar statistics.

    Purpose:
        Provides a post-computation sanity check that the response map is
        numerically clean and within the theoretical [-1, 1] ZNCC range.

    Inputs:
        response_map: 2D float32 ZNCC output from compute_zncc_fft or
                      compute_zncc_spatial.

    Outputs:
        Tuple (max_score, mean_score) as Python floats.

    Raises:
        MatchingError: If the response map contains NaN/Inf or is empty.

    Complexity: O(N) where N = number of valid positions.

    Numerical Notes:
        ZNCC scores slightly outside [-1, 1] may arise from floating-point
        rounding; this is expected and acceptable. Extreme deviations (> 1.5)
        indicate numerical failure upstream and raise MatchingError.

    Industrial Notes:
        Called by the inference orchestrator to gate fallback logic. A
        max_score below 0.3 on a nominal image is a strong signal of a
        catastrophic mismatch or a featureless reference patch.
    """
    if response_map.size == 0:
        raise MatchingError("Response map is empty.")

    _check_no_nan_inf(response_map, "response_map")

    max_score = float(np.max(response_map))
    mean_score = float(np.mean(response_map))

    if max_score > 1.5 or max_score < -1.5:
        raise MatchingError(
            f"Response map max value {max_score:.4f} is outside the stable ZNCC range. "
            "Check that inputs were correctly normalised before calling this function."
        )

    logger.debug("Response map: max=%.4f  mean=%.4f", max_score, mean_score)
    return max_score, mean_score


def compute_zncc_fft(
    template: ImageArray,
    search: ImageArray,
) -> ImageArray:
    """
    Computes the Zero Mean Normalized Cross-Correlation response map using
    the FFT convolution theorem.

    This is the PRODUCTION implementation used for all inference runs.

    Purpose:
        Locates where in the search image the given template correlates
        maximally. Returns a dense map of ZNCC scores; coordinate selection
        is performed exclusively by downstream modules.

    Inputs:
        template: 2D float32 zero-mean normalised reference patch.
        search:   2D float32 zero-mean normalised search image.
                  Must be strictly larger than template in both dimensions.

    Outputs:
        response_map: 2D float32 array of shape
                      (s_h - t_h + 1, s_w - t_w + 1).
                      Values nominally in [-1, 1].

    Raises:
        PreprocessingError: On dtype or shape violations.
        MatchingError:      On NaN/Inf inputs or numerical instability.

    Complexity:
        Time:   O((N_s + N_t) log(N_s + N_t)) via FFT convolution.
        Memory: O(N_s) for complex-valued FFT buffers (float64 internally).

    Numerical Notes:
        - FFT cross-correlation is computed in float64 to prevent catastrophic
          cancellation in the complex multiplications, then cast back to float32.
        - Local variance is clamped to [0, ∞) before sqrt.
        - Denominator is lower-bounded by _EPSILON.
        - Template is zero-mean normalised before correlation.

    Industrial Notes:
        For FinFET structures with high periodicity, the response map will
        contain multiple peaks of similar height. Peak selection (NMS +
        tie-breaking) is the downstream responsibility — this function returns
        the complete map to preserve all information.
    """
    _validate_zncc_inputs(template, search)

    std = float(template.std())
    if std < _EPSILON:
        t_h, t_w = template.shape
        s_h, s_w = search.shape
        return np.zeros((s_h - t_h + 1, s_w - t_w + 1), dtype=np.float32)

    response_map = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)

    # Clip to guard against extreme floating-point excursions.
    response_map = np.clip(response_map, -1.0, 1.0)

    logger.debug(
        "compute_zncc_fft (optimized): response map shape=%s  max=%.4f",
        response_map.shape, float(np.max(response_map)),
    )
    return response_map


def compute_zncc_spatial(
    template: ImageArray,
    search: ImageArray,
) -> ImageArray:
    """
    Computes Zero Mean Normalized Cross-Correlation using a direct spatial
    sliding-window approach.

    This is the REFERENCE/DEBUGGING implementation only.
    It is NOT used in production inference due to its O(N_s * N_t) complexity.

    Purpose:
        Provides a numerically independent ground truth to validate the FFT
        implementation on small images during unit testing and ablation studies.

    Inputs:
        template: 2D float32 zero-mean normalised reference patch.
        search:   2D float32 zero-mean normalised search image.
                  Must be strictly larger than template in both dimensions.

    Outputs:
        response_map: 2D float32 array of shape
                      (s_h - t_h + 1, s_w - t_w + 1).
                      Values nominally in [-1, 1].

    Raises:
        PreprocessingError: On dtype or shape violations.
        MatchingError:      On NaN/Inf inputs.

    Complexity:
        Time:   O(N_s * N_t) — prohibitive for large images.
        Memory: O(N_t) per window evaluation.

    Numerical Notes:
        - Each window and the template are independently zero-mean subtracted
          to minimise floating-point cancellation at small scales.
        - Variance is clamped to max(var, 0) before sqrt.

    Industrial Notes:
        Maximum recommended image size: 200×200 search, 20×20 template.
        Beyond this, runtime becomes impractical. Use compute_zncc_fft()
        for all production loads.
    """
    _validate_zncc_inputs(template, search)

    t_h, t_w = template.shape
    s_h, s_w = search.shape
    valid_h = s_h - t_h + 1
    valid_w = s_w - t_w + 1

    tmpl_zm = normalize_template(template)
    tmpl_std = float(tmpl_zm.std())
    n_pixels = float(t_h * t_w)

    response_map = np.full((valid_h, valid_w), -1.0, dtype=np.float32)

    for row in range(valid_h):
        for col in range(valid_w):
            window = search[row : row + t_h, col : col + t_w].astype(np.float64)
            win_mean = float(window.mean())
            win_zm = (window - win_mean).astype(np.float32)

            win_var = float(np.maximum(np.mean(win_zm ** 2), 0.0))
            win_std = float(np.sqrt(win_var))

            denom = n_pixels * tmpl_std * win_std
            if denom < _EPSILON:
                response_map[row, col] = 0.0
            else:
                numerator = float(np.sum(tmpl_zm.astype(np.float64) * win_zm.astype(np.float64)))
                response_map[row, col] = np.clip(numerator / denom, -1.0, 1.0)

    logger.debug(
        "compute_zncc_spatial: response map shape=%s  max=%.4f",
        response_map.shape, float(np.max(response_map)),
    )
    return response_map
