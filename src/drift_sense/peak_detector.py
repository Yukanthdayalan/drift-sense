"""
Peak detection module for the Drift-Sense inference engine.

Receives the ZNCC response map produced by zncc.py and executes a four-stage
deterministic pipeline to return the single most reliable localisation candidate.

ARCHITECTURAL CONTRACT:
  - This module does NOT compute ZNCC, perform scale search, or do subpixel refinement.
  - All decisions are deterministic — no randomness, no ML.
  - The tie-break rule matches the competition specification exactly:
      if (top1.score - top2.score) > delta  →  return top1 unconditionally
      else                                   →  return the candidate nearest to center
  - The response surface is NEVER modified or bias-weighted.

Stages:
  1. Morphological local-maximum extraction (O(N) via dilation).
  2. Response-delta threshold to reject weak peaks.
  3. Candidate construction and sorting (score desc, distance asc).
  4. Deterministic tie-break selection.
"""
import math
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from drift_sense.config import NMSConfig, TieBreakConfig
from drift_sense.exceptions import PeakDetectionError
from drift_sense.logging_utils import get_logger
from drift_sense.types import ImageArray

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class PeakCandidate:
    """
    Immutable, fully comparable representation of a single localisation candidate.

    Fields are declared in comparison-priority order so that the auto-generated
    ordering (order=True) naturally sorts by score descending then distance
    ascending when used with a sign-flip key.

    Attributes:
        x:                  Integer column index in the ZNCC response map.
        y:                  Integer row index in the ZNCC response map.
        score:              ZNCC score at this position, nominally in [-1, 1].
        distance_to_center: Euclidean distance from (x, y) to the map centre.
    """
    x: int
    y: int
    score: float
    distance_to_center: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_response_map(response_map: ImageArray) -> None:
    """
    Validates that the response map meets structural requirements before processing.

    Args:
        response_map: The 2D float32 ZNCC response map to validate.

    Raises:
        PeakDetectionError: If the map is empty, not 2D, not float32, or contains
                            NaN or Inf values.
    """
    if not isinstance(response_map, np.ndarray):
        raise PeakDetectionError(
            f"response_map must be a numpy ndarray, got {type(response_map).__name__}."
        )
    if response_map.size == 0:
        raise PeakDetectionError("response_map is empty (size 0).")
    if response_map.ndim != 2:
        raise PeakDetectionError(
            f"response_map must be 2D, got {response_map.ndim}D."
        )
    if response_map.dtype != np.float32:
        raise PeakDetectionError(
            f"response_map must be float32, got {response_map.dtype}."
        )
    if not np.isfinite(response_map).all():
        raise PeakDetectionError(
            "response_map contains NaN or Inf values. "
            "Ensure zncc.py produced a clean output before calling peak_detector."
        )


def _build_nms_kernel(radius: int) -> np.ndarray:
    """
    Builds an elliptical (approximately circular) morphological structuring element.

    Args:
        radius: Neighbourhood radius in pixels. Kernel side = 2*radius + 1.

    Returns:
        uint8 structuring element array suitable for cv2.dilate.
    """
    side = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))


def _compute_distance(x: int, y: int, cx: float, cy: float) -> float:
    """
    Computes Euclidean distance from pixel (x, y) to the map centre (cx, cy).

    Args:
        x:  Column index.
        y:  Row index.
        cx: Centre column (float for sub-pixel precision).
        cy: Centre row (float for sub-pixel precision).

    Returns:
        float: Euclidean distance in pixels.
    """
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def _apply_morphological_nms(
    response_map: ImageArray,
    radius: int,
) -> np.ndarray:
    """
    Computes a boolean mask of strict local maxima via morphological dilation.

    A pixel at (r, c) is a local maximum iff its value equals the maximum of
    all values in the elliptical neighbourhood of given radius. This is
    equivalent to: response_map == dilate(response_map, kernel).

    Since cv2.dilate propagates exact float values without arithmetic, the
    equality check is numerically exact (no floating-point accumulation error).

    Complexity: O(N * radius²) amortised — O(N) for large maps via separable
    approximation in OpenCV's internal implementation.

    Args:
        response_map: 2D float32 ZNCC response map.
        radius:       Neighbourhood radius for suppression.

    Returns:
        Boolean mask, True where response_map is a strict local maximum.
    """
    kernel = _build_nms_kernel(radius)
    dilated = cv2.dilate(response_map, kernel)
    # Exact equality is safe here: dilate only copies values, no arithmetic.
    return response_map == dilated


def _score_threshold(max_response: float, response_delta: float) -> float:
    """
    Computes the minimum admissible score from the max response and delta.

    Args:
        max_response:   Maximum value in the response map.
        response_delta: Allowable score gap below the maximum.

    Returns:
        float: Minimum score a candidate must meet to survive thresholding.
    """
    return max_response - response_delta


def _build_candidates(
    response_map: ImageArray,
    local_max_mask: np.ndarray,
    threshold: float,
    cx: float,
    cy: float,
) -> List[PeakCandidate]:
    """
    Converts surviving local-maximum pixels into PeakCandidate objects.

    Args:
        response_map:  2D float32 ZNCC response map.
        local_max_mask: Boolean mask from morphological NMS.
        threshold:     Minimum score to keep.
        cx:            Map centre column (float).
        cy:            Map centre row (float).

    Returns:
        List of PeakCandidate objects for all pixels passing both NMS and threshold.
    """
    # Combine NMS mask with threshold in one vectorised pass.
    above_threshold = response_map >= threshold
    accepted = local_max_mask & above_threshold

    rows, cols = np.where(accepted)
    candidates: List[PeakCandidate] = []

    for r, c in zip(rows.tolist(), cols.tolist()):
        score = float(response_map[r, c])
        dist = _compute_distance(int(c), int(r), cx, cy)
        candidates.append(PeakCandidate(x=int(c), y=int(r), score=score, distance_to_center=dist))

    return candidates


def _sort_candidates(candidates: List[PeakCandidate]) -> List[PeakCandidate]:
    """
    Sorts candidates by descending ZNCC score, then ascending distance to center.

    This ordering ensures that:
      - The strongest match is always first.
      - Among equal-score candidates, the center-nearest is first.
      - The sort is stable across identical inputs (deterministic).

    Args:
        candidates: Unsorted list of PeakCandidate objects.

    Returns:
        Sorted list (new object, original list unchanged).
    """
    return sorted(candidates, key=lambda c: (-c.score, c.distance_to_center))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_peaks(
    response_map: ImageArray,
    nms_config: NMSConfig,
) -> List[PeakCandidate]:
    """
    Executes Stages 1–3: morphological NMS, threshold filtering, candidate construction,
    and deterministic sorting of the ZNCC response map.

    Args:
        response_map: 2D float32 ZNCC response map, shape (H, W).
        nms_config:   Immutable NMS configuration parameters.

    Returns:
        Sorted list of PeakCandidate objects (score desc, distance asc).
        Contains at least one entry (the global maximum is always kept).

    Raises:
        PeakDetectionError: If the response map is invalid, degenerate, or produces
                            no candidates above the configured threshold.

    Complexity:
        Time:   O(N) for morphological dilation + O(K log K) for sort,
                where K << N is the number of surviving peaks.
        Memory: O(N) for the dilated map + O(K) for the candidate list.

    Numerical Notes:
        - The flat-map edge case is handled: if every pixel has the same value,
          every pixel is a local max. Threshold = max - delta = max - delta, so only
          pixels within delta of max survive. For a perfectly flat map, all pixels
          survive NMS and threshold. The _sort_candidates step then places the
          center-nearest pixel first due to the secondary distance key.
        - np.where is used instead of Python loops for O(1) per-pixel vectorisation.

    Industrial Notes:
        For periodic FinFET structures, this stage is expected to return multiple
        candidates with near-identical scores. The tie-breaking in select_best_peak
        resolves the ambiguity using center distance, matching the competition rule.
    """
    _validate_response_map(response_map)

    h, w = response_map.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    max_response = float(np.max(response_map))
    threshold = _score_threshold(max_response, nms_config.response_delta)

    logger.debug(
        "detect_peaks: map=%dx%d  max=%.4f  threshold=%.4f  radius=%d",
        w, h, max_response, threshold, nms_config.suppression_radius,
    )

    local_max_mask = _apply_morphological_nms(response_map, nms_config.suppression_radius)
    candidates = _build_candidates(response_map, local_max_mask, threshold, cx, cy)

    if not candidates:
        raise PeakDetectionError(
            f"No peaks found above threshold {threshold:.4f} "
            f"(max_response={max_response:.4f}, response_delta={nms_config.response_delta})."
        )

    sorted_candidates = _sort_candidates(candidates)
    logger.info(
        "detect_peaks: %d candidate(s) found. Top score=%.4f at (%d, %d).",
        len(sorted_candidates),
        sorted_candidates[0].score,
        sorted_candidates[0].x,
        sorted_candidates[0].y,
    )
    return sorted_candidates


def select_best_peak(
    candidates: List[PeakCandidate],
    response_shape: Tuple[int, int],
    tie_config: TieBreakConfig,
) -> PeakCandidate:
    """
    Executes Stage 4: deterministic tie-breaking to select the single best candidate.

    Algorithm (matches competition specification exactly):
      Let top1 = candidates[0], top2 = candidates[1] (if it exists).
      If (top1.score - top2.score) > delta:
          Return top1 immediately. No center bias applied.
      Else:
          Collect all candidates C_tied where top1.score - C.score <= delta.
          Return argmin over C_tied of distance_to_center.

    The response surface is NEVER modified. No Gaussian weighting is applied.
    The tie-break rule activates ONLY when scores are genuinely ambiguous.

    Args:
        candidates:     Non-empty sorted list from detect_peaks (score desc).
        response_shape: (H, W) of the original ZNCC response map, used only for
                        logging context.
        tie_config:     Immutable tie-breaking configuration.

    Returns:
        The single winning PeakCandidate.

    Raises:
        PeakDetectionError: If candidates list is empty.

    Complexity:
        Time:   O(K) where K = number of tied candidates (K << N).
        Memory: O(K).

    Numerical Notes:
        Score comparison uses (top1.score - candidate.score) <= delta with strict
        float subtraction. np.isclose is not used here because the delta threshold
        represents an engineering tolerance, not a floating-point epsilon.
    """
    if not candidates:
        raise PeakDetectionError("select_best_peak received an empty candidate list.")

    top1 = candidates[0]

    # Only one candidate — return immediately, no tie-break needed.
    if len(candidates) == 1:
        logger.info(
            "select_best_peak: single candidate at (%d, %d) score=%.4f.",
            top1.x, top1.y, top1.score,
        )
        return top1

    top2 = candidates[1]
    score_gap = top1.score - top2.score

    if score_gap > tie_config.delta:
        # Clear winner — return immediately without any center bias.
        logger.info(
            "select_best_peak: clear winner at (%d, %d) score=%.4f gap=%.4f > delta=%.4f.",
            top1.x, top1.y, top1.score, score_gap, tie_config.delta,
        )
        return top1

    # Tie detected: collect all candidates within delta of top1.
    tied: List[PeakCandidate] = [
        c for c in candidates
        if (top1.score - c.score) <= tie_config.delta
    ]

    # Among tied candidates, choose the one nearest to the image center.
    winner = min(tied, key=lambda c: c.distance_to_center)

    logger.info(
        "select_best_peak: TIE-BREAK activated. %d tied candidates. "
        "Winner at (%d, %d) score=%.4f dist=%.2f.",
        len(tied), winner.x, winner.y, winner.score, winner.distance_to_center,
    )
    return winner


def detect_best_peak(
    response_map: ImageArray,
    nms_config: NMSConfig,
    tie_config: TieBreakConfig,
) -> PeakCandidate:
    """
    Full four-stage pipeline: detect all peaks and select the single best candidate.

    This is the primary public entry point for the inference orchestrator.

    Args:
        response_map: 2D float32 ZNCC response map.
        nms_config:   NMS configuration (radius, response_delta).
        tie_config:   Tie-breaking configuration (delta).

    Returns:
        The single winning PeakCandidate with integer (x, y) coordinates.

    Raises:
        PeakDetectionError: On any validation failure or if no valid peak is found.

    Complexity:
        Time:   O(N log N) worst case (dilation + sort). O(N) typical.
        Memory: O(N).
    """
    t_start = time.perf_counter()

    candidates = detect_peaks(response_map, nms_config)
    winner = select_best_peak(candidates, response_map.shape, tie_config)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        "detect_best_peak: completed in %.2f ms. Winner: (%d, %d) score=%.4f.",
        elapsed_ms, winner.x, winner.y, winner.score,
    )
    return winner


def get_sharpness(response_map: ImageArray, x: int, y: int) -> float:
    """
    Computes peak sharpness as the difference between the center peak
    and the mean of the 3x3 surrounding annulus.
    """
    h, w = response_map.shape
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return 0.0
        
    roi = response_map[y-1:y+2, x-1:x+2]
    total_sum = float(np.sum(roi))
    center_val = float(roi[1, 1])
    annulus_mean = (total_sum - center_val) / 8.0
    return center_val - annulus_mean
