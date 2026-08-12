"""
Scale search module for the Drift-Sense inference engine.

Implements:
  - Optional PSD-based scale prior (heuristic accelerator only)
  - Coarse scale grid search via ZNCC score evaluation
  - Fine scale grid refinement around the coarse winner

ARCHITECTURAL CONTRACT:
  PSD is ONLY a heuristic. It may narrow the coarse search bracket.
  PSD NEVER determines the final scale. The ZNCC surface is always authoritative.
"""
import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np

from drift_sense.config import ScaleSearchConfig
from drift_sense.exceptions import ScaleSearchError
from drift_sense.logging_utils import get_logger
from drift_sense.types import ImageArray

logger: logging.Logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PSD Prior (Optional Heuristic)
# ---------------------------------------------------------------------------

def _compute_psd_scale_prior(
    ref_image: ImageArray,
    search_image: ImageArray,
) -> Tuple[Optional[float], float]:
    """
    Estimates the scale ratio between search and reference images using 1D Power
    Spectral Density peak analysis on the horizontal axis projection.

    The fundamental spatial frequency of a periodic FinFET/DRAM structure will
    scale linearly with magnification. The ratio of peak frequencies gives a
    direct analytical estimate of the scale factor.

    Complexity: O(N log N) per image for rfft.

    Args:
        ref_image:    Float32 normalised reference image.
        search_image: Float32 normalised search image.

    Returns:
        Tuple of (scale_estimate, confidence):
          - scale_estimate: Estimated scale, or None if estimation failed.
          - confidence:     Peak prominence in units of standard deviations.
    """
    def _dominant_frequency(image: ImageArray) -> Tuple[Optional[float], float]:
        """Return (dominant_freq_normalised, prominence_sigma) for the horizontal PSD."""
        h, w = image.shape[:2]
        # Apply 1D Hanning window along the horizontal axis to suppress spectral leakage.
        window = np.hanning(w).astype(np.float32)
        windowed = image * window[np.newaxis, :]

        # Sum rows → 1D horizontal projection, then rfft for the 1-sided spectrum.
        row_projection = windowed.sum(axis=0)
        spectrum = np.abs(np.fft.rfft(row_projection)) ** 2

        # Ignore DC component (index 0) and enforce a minimum searchable frequency.
        min_freq_idx = max(1, w // 64)
        search_spectrum = spectrum[min_freq_idx:]

        if search_spectrum.size == 0:
            return None, 0.0

        peak_local_idx = int(np.argmax(search_spectrum))
        peak_global_idx = peak_local_idx + min_freq_idx
        peak_freq_normalised = peak_global_idx / w

        # Prominence: how many sigma above the mean the peak sits.
        mean_s = float(np.mean(search_spectrum))
        std_s = float(np.std(search_spectrum))
        if std_s < 1e-12:
            return None, 0.0

        prominence = (float(search_spectrum[peak_local_idx]) - mean_s) / std_s
        return peak_freq_normalised, prominence

    ref_freq, ref_conf = _dominant_frequency(ref_image)
    search_freq, search_conf = _dominant_frequency(search_image)

    if ref_freq is None or search_freq is None:
        logger.debug("PSD prior: dominant frequency undefined — skipping prior.")
        return None, 0.0

    if ref_freq < 1e-9:
        logger.debug("PSD prior: reference dominant frequency is near-zero — skipping prior.")
        return None, 0.0

    scale_estimate = search_freq / ref_freq
    confidence = min(ref_conf, search_conf)

    logger.debug(
        "PSD prior: ref_freq=%.6f  search_freq=%.6f  scale_est=%.3f  confidence=%.2f",
        ref_freq, search_freq, scale_estimate, confidence,
    )
    return scale_estimate, confidence


# ---------------------------------------------------------------------------
# ZNCC Proxy (fast, coarse — used for scale scoring only)
# ---------------------------------------------------------------------------

def _score_scale_candidate(
    ref_image: ImageArray,
    search_image: ImageArray,
    scale: float,
) -> float:
    """
    Resizes the reference to the given scale, then evaluates a fast normalised
    cross-correlation proxy score using cv2.matchTemplate (TM_CCOEFF_NORMED).

    This is used ONLY for scale scoring during coarse/fine grid search.
    The full FFT-ZNCC engine in zncc.py is the authoritative matcher.

    Complexity: O(N_s * N_r) worst-case via OpenCV's optimised implementation.

    Args:
        ref_image:    Float32 normalised reference image.
        search_image: Float32 normalised search image.
        scale:        Candidate scale factor.

    Returns:
        float: Maximum NCC score in [−1, 1]. Returns −1.0 on dimensional failure.
    """
    ref_h, ref_w = ref_image.shape[:2]
    scaled_h = max(1, round(ref_h * scale))
    scaled_w = max(1, round(ref_w * scale))

    search_h, search_w = search_image.shape[:2]
    if scaled_h >= search_h or scaled_w >= search_w:
        logger.debug("Scale %.3f: scaled template (%dx%d) exceeds search image — skipping.", scale, scaled_w, scaled_h)
        return -1.0

    scaled_ref = cv2.resize(
        ref_image,
        (scaled_w, scaled_h),
        interpolation=cv2.INTER_LINEAR,
    )

    result = cv2.matchTemplate(search_image, scaled_ref, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


# ---------------------------------------------------------------------------
# Coarse Search
# ---------------------------------------------------------------------------

def coarse_scale_search(
    ref_image: ImageArray,
    search_image: ImageArray,
    config: ScaleSearchConfig,
    psd_center: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Evaluates a uniform coarse grid of scale candidates and returns the best.

    If a valid PSD prior is supplied, the search bracket is centred around that
    prior while remaining within [min_scale, max_scale]. If no PSD prior is
    available, the full configured range is searched uniformly.

    Complexity: O(K * N_s * N_r) where K = number of coarse grid points.

    Args:
        ref_image:    Float32 normalised reference image.
        search_image: Float32 normalised search image.
        config:       Immutable scale search configuration.
        psd_center:   Optional PSD-derived scale estimate (heuristic only).

    Returns:
        Tuple of (best_scale, best_score).

    Raises:
        ScaleSearchError: If no valid candidate was found across the entire grid.
    """
    # Build the coarse grid, bounded strictly to [min_scale, max_scale].
    if psd_center is not None:
        bracket_half = (config.max_scale - config.min_scale) / 2.0
        low  = max(config.min_scale, psd_center - bracket_half / 2.0)
        high = min(config.max_scale, psd_center + bracket_half / 2.0)
        logger.debug("Coarse search: PSD-narrowed bracket [%.2f, %.2f].", low, high)
    else:
        low, high = config.min_scale, config.max_scale
        logger.debug("Coarse search: full bracket [%.2f, %.2f].", low, high)

    n_steps = max(2, round((high - low) / config.coarse_step) + 1)
    grid: List[float] = list(np.linspace(low, high, n_steps))

    best_scale = -1.0
    best_score = -2.0
    found_valid = False

    for scale in grid:
        score = _score_scale_candidate(ref_image, search_image, scale)
        logger.debug("Coarse  scale=%.3f  score=%.4f", scale, score)
        if score > -1.0 and score > best_score:
            best_score = score
            best_scale = scale
            found_valid = True

    if not found_valid:
        raise ScaleSearchError(
            f"Coarse scale search found no valid candidate in [{low:.2f}, {high:.2f}]."
        )

    logger.info("Coarse search winner: scale=%.3f  score=%.4f", best_scale, best_score)
    return best_scale, best_score


# ---------------------------------------------------------------------------
# Fine Search
# ---------------------------------------------------------------------------

def fine_scale_search(
    ref_image: ImageArray,
    search_image: ImageArray,
    coarse_scale: float,
    config: ScaleSearchConfig,
) -> Tuple[float, float]:
    """
    Refines scale around the coarse winner using a dense fine grid.

    The fine search bracket is ±1 coarse step around the coarse winner,
    always clamped to [min_scale, max_scale].

    Complexity: O(K_fine * N_s * N_r).

    Args:
        ref_image:    Float32 normalised reference image.
        search_image: Float32 normalised search image.
        coarse_scale: Best scale from the coarse pass.
        config:       Immutable scale search configuration.

    Returns:
        Tuple of (best_fine_scale, best_fine_score).

    Raises:
        ScaleSearchError: If fine search yields no valid candidate.
    """
    low  = max(config.min_scale, coarse_scale - config.coarse_step)
    high = min(config.max_scale, coarse_scale + config.coarse_step)

    n_steps = max(2, round((high - low) / config.fine_step) + 1)
    grid: List[float] = list(np.linspace(low, high, n_steps))

    best_scale = -1.0
    best_score = -2.0
    found_valid = False

    for scale in grid:
        score = _score_scale_candidate(ref_image, search_image, scale)
        logger.debug("Fine  scale=%.3f  score=%.4f", scale, score)
        if score > -1.0 and score > best_score:
            best_score = score
            best_scale = scale
            found_valid = True

    if not found_valid:
        raise ScaleSearchError(
            f"Fine scale search found no valid candidate around coarse={coarse_scale:.3f}."
        )

    logger.info("Fine search winner: scale=%.3f  score=%.4f", best_scale, best_score)
    return best_scale, best_score


# ---------------------------------------------------------------------------
# Public Orchestrator
# ---------------------------------------------------------------------------

def ultra_fine_scale_search(
    ref_image: ImageArray,
    search_image: ImageArray,
    fine_scale: float,
    config: ScaleSearchConfig,
) -> Tuple[float, float]:
    """
    Ultra-fine scale refinement around the fine search winner.

    The ultra-fine search bracket is ±1 fine step around the fine winner,
    always clamped to [min_scale, max_scale].

    Args:
        ref_image:    Float32 normalised reference image.
        search_image: Float32 normalised search image.
        fine_scale:   Best scale from the fine pass.
        config:       Immutable scale search configuration.

    Returns:
        Tuple of (best_ultra_fine_scale, best_ultra_fine_score).

    Raises:
        ScaleSearchError: If ultra-fine search yields no valid candidate.
    """
    low  = max(config.min_scale, fine_scale - config.fine_step)
    high = min(config.max_scale, fine_scale + config.fine_step)

    n_steps = max(2, round((high - low) / config.ultra_fine_step) + 1)
    grid: List[float] = list(np.linspace(low, high, n_steps))

    best_scale = -1.0
    best_score = -2.0
    found_valid = False

    for scale in grid:
        score = _score_scale_candidate(ref_image, search_image, scale)
        logger.debug("Ultra-fine  scale=%.4f  score=%.4f", scale, score)
        if score > -1.0 and score > best_score:
            best_score = score
            best_scale = scale
            found_valid = True

    if not found_valid:
        raise ScaleSearchError(
            f"Ultra-fine scale search found no valid candidate around fine={fine_scale:.3f}."
        )

    logger.info("Ultra-fine search winner: scale=%.4f  score=%.4f", best_scale, best_score)
    return best_scale, best_score


def estimate_scale(
    ref_image: ImageArray,
    search_image: ImageArray,
    config: ScaleSearchConfig,
) -> Tuple[float, float]:
    """
    Full scale estimation pipeline: PSD prior → Coarse Search → Fine Search → Ultra-Fine Search.

    ARCHITECTURAL CONTRACT (enforced here):
      - PSD result is used ONLY to optionally narrow the coarse bracket.
      - If PSD confidence is below threshold, it is silently discarded.
      - The final scale is always determined by the ZNCC-proxy scoring surface.

    Args:
        ref_image:    Float32 normalised reference image.
        search_image: Float32 normalised search image.
        config:       Immutable scale search configuration.

    Returns:
        Tuple of (final_scale, final_score).

    Raises:
        ScaleSearchError: If both coarse and fine search fail.
    """
    # --- Optional PSD prior ---
    psd_center: Optional[float] = None

    if config.use_psd_prior:
        psd_est, psd_conf = _compute_psd_scale_prior(ref_image, search_image)

        if psd_est is not None and psd_conf >= config.psd_confidence_threshold:
            if config.min_scale <= psd_est <= config.max_scale:
                psd_center = psd_est
                logger.info("PSD prior accepted: scale=%.3f  confidence=%.2f", psd_est, psd_conf)
            else:
                logger.info(
                    "PSD prior out-of-bounds (%.3f not in [%.1f, %.1f]) — discarded.",
                    psd_est, config.min_scale, config.max_scale,
                )
        else:
            logger.info("PSD prior discarded: low confidence (%.2f < %.2f).", psd_conf, config.psd_confidence_threshold)

    # --- Coarse search ---
    coarse_scale, coarse_score = coarse_scale_search(
        ref_image, search_image, config, psd_center=psd_center
    )

    # --- Fine search ---
    fine_scale, fine_score = fine_scale_search(
        ref_image, search_image, coarse_scale, config
    )

    # --- Ultra-fine search ---
    final_scale, final_score = ultra_fine_scale_search(
        ref_image, search_image, fine_scale, config
    )

    return final_scale, final_score


def estimate_top_n_scales(
    ref_image: ImageArray,
    search_image: ImageArray,
    config: ScaleSearchConfig,
    n_scales: int = 2
) -> List[float]:
    """
    Returns the top N fully-refined scales.
    Matches the V2 PSD/coarse/fine/ultra-fine logic but branches at the coarse level.
    """
    psd_center: Optional[float] = None
    if config.use_psd_prior:
        psd_est, psd_conf = _compute_psd_scale_prior(ref_image, search_image)
        if psd_est is not None and psd_conf >= config.psd_confidence_threshold:
            if config.min_scale <= psd_est <= config.max_scale:
                psd_center = psd_est
                
    if psd_center is not None:
        bracket_half = (config.max_scale - config.min_scale) / 2.0
        low  = max(config.min_scale, psd_center - bracket_half / 2.0)
        high = min(config.max_scale, psd_center + bracket_half / 2.0)
    else:
        low, high = config.min_scale, config.max_scale

    n_steps = max(2, round((high - low) / config.coarse_step) + 1)
    grid = list(np.linspace(low, high, n_steps))

    coarse = []
    for scale in grid:
        score = _score_scale_candidate(ref_image, search_image, scale)
        if score > -1.0:
            coarse.append((scale, score))
            
    if not coarse:
        raise ScaleSearchError("No valid candidate found in top-n coarse search.")
        
    coarse.sort(key=lambda x: x[1], reverse=True)
    top_coarse = [x[0] for x in coarse[:n_scales]]
    
    final_scales = []
    for cs in top_coarse:
        try:
            fine_scale, _ = fine_scale_search(ref_image, search_image, cs, config)
            ultra_scale, _ = ultra_fine_scale_search(ref_image, search_image, fine_scale, config)
            final_scales.append(ultra_scale)
        except ScaleSearchError:
            continue
            
    if not final_scales:
        raise ScaleSearchError("No valid candidate found after refinement.")
        
    return final_scales
