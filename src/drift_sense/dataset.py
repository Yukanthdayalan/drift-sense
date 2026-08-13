"""
Synthetic FinFET dataset generator for the Drift-Sense inference engine.

Generates realistic semiconductor wafer/SEM inspection image pairs with
controlled ground truth for benchmarking the localization pipeline.

Coordinate Convention:
    Ground truth exposes BOTH top-left and center coordinates.

    - gt_top_left_(x,y): the top-left corner of the scaled reference region
      in the search image coordinate space.
    - gt_center_(x,y): the center of the scaled reference region, computed as:
        gt_center_x = gt_top_left_x + (scaled_w - 1) / 2
        gt_center_y = gt_top_left_y + (scaled_h - 1) / 2

    The inference engine (matcher.py) returns CENTER coordinates using the
    same formula.  Direct comparison is therefore valid.

Geometric Relationship:
    reference_size=20, nominal_scale=10 → scaled template ≈ 200×200
    This fits comfortably in a 1000×1000 search image and matches the
    dimensions used by the rest of the test suite.

Disambiguation Strategy:
    The periodic FinFET structure naturally creates many near-identical ZNCC
    peaks.  To make the correct location identifiable, the generator injects
    realistic process defects (missing fins, gate breaks, line-width
    variation, defect clusters, edge roughness) concentrated inside the
    target region.  These are NOT artificial markers — they resemble real
    semiconductor manufacturing defects.
"""
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Configuration for the synthetic FinFET sample generator.

    Default geometry:
        ref_size=20 at scale≈10 → scaled template ≈ 200×200 in 1000×1000.
    """
    # Image dimensions
    search_size: int = 1000
    ref_size: int = 20

    # Scale range (consistent with ScaleSearchConfig: min_scale=9.0, max_scale=11.0)
    scale_min: float = 9.5
    scale_max: float = 10.5

    # FinFET structural parameters (in search-image pixels)
    fin_period: float = 14.0       # Vertical line/space period
    gate_period: float = 20.0      # Horizontal gate period
    fin_width_frac: float = 0.45   # Fraction of fin_period occupied by fin
    gate_width_frac: float = 0.35  # Fraction of gate_period occupied by gate
    fin_intensity: float = 190.0   # Fin brightness
    gate_intensity: float = 170.0  # Gate brightness
    background_intensity: float = 50.0  # Background brightness
    intersection_boost: float = 20.0  # Extra brightness at fin-gate crossings

    # Manufacturing variation (low-frequency spatial noise)
    mfg_variation_sigma: float = 8.0

    # Target-region defect parameters
    num_target_defects: int = 20      # Number of defects inside target region
    defect_min_size: int = 3          # Minimum defect size in pixels
    defect_max_size: int = 30         # Maximum defect size in pixels
    defect_intensity_low: float = 30.0   # Defect can darken to this
    defect_intensity_high: float = 220.0  # Defect can brighten to this

    # Global (background) defects — scattered across the whole image
    num_global_defects: int = 6
    global_defect_min_size: int = 3
    global_defect_max_size: int = 15

    # Noise parameters (applied independently to ref and search)
    noise_sigma_search: float = 8.0
    noise_sigma_ref: float = 5.0

    # Brightness / contrast jitter for reference
    brightness_jitter: float = 5.0
    contrast_jitter: float = 0.05

    # Illumination gradient magnitude
    illumination_gradient_max: float = 15.0

    # Placement margin (minimum distance from border)
    placement_margin: int = 20

    # Quality gate
    min_variance: float = 10.0

    # New additions for compliance
    speckle_sigma: float = 0.05
    blur_sigma_search_min: float = 0.5
    blur_sigma_search_max: float = 2.0
    blur_sigma_ref_min: float = 0.0
    blur_sigma_ref_max: float = 0.5
    rotation_range: float = 15.0  # Max rotation in degrees
    edge_boost_factor: float = 0.3 # Factor for edge brightening


@dataclass
class SampleMetadata:
    """Ground-truth metadata for a single generated sample.

    Exposes both top-left and center coordinates for the target region
    in search-image pixel coordinates.
    """
    gt_top_left_x: int
    gt_top_left_y: int
    gt_center_x: float
    gt_center_y: float
    scaled_width: int
    scaled_height: int
    scale: float
    seed: int
    ref_width: int
    ref_height: int
    config: Dict[str, Any] = field(default_factory=dict)


class InvalidSampleError(Exception):
    """Raised when a generated sample fails quality checks."""
    pass


# ---------------------------------------------------------------------------
# FinFET Pattern Synthesis
# ---------------------------------------------------------------------------

def _generate_finfet_base(
    h: int,
    w: int,
    rng: Generator,
    config: GeneratorConfig,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    fin_period: float = None,
    gate_period: float = None,
) -> np.ndarray:
    """
    Synthesise a base FinFET wafer layout as a float64 image.

    The pattern contains:
    - Vertical periodic fin structures (trapezoidal profile)
    - Horizontal periodic gate structures
    - Fin-gate intersection brightening
    - Low-frequency manufacturing variation
    - Subtle per-element intensity jitter

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        rng: Deterministic NumPy random generator.
        config: Generator configuration.
        x_offset: Horizontal phase offset.
        y_offset: Vertical phase offset.
        fin_period: Override for fin_period.
        gate_period: Override for gate_period.

    Returns:
        float64 image of shape (h, w), values approximately in [0, 255].
    """
    if fin_period is None:
        fin_period = config.fin_period
    if gate_period is None:
        gate_period = config.gate_period

    image = np.full((h, w), config.background_intensity, dtype=np.float64)

    # --- Vertical fins ---
    fin_half = fin_period * config.fin_width_frac / 2.0
    x_coords = np.arange(w, dtype=np.float64)
    # Distance from nearest fin center
    fin_phase = np.mod(x_offset + x_coords, fin_period)
    fin_center_dist = np.abs(fin_phase - fin_period / 2.0)
    # Smooth fin profile: bright where close to center, dark in space
    fin_mask = np.clip(1.0 - (fin_center_dist - fin_half) / 2.0, 0.0, 1.0)
    # Per-fin jitter
    num_fins = int(np.ceil((w + x_offset) / fin_period))
    fin_jitter = rng.uniform(-4.0, 4.0, num_fins)
    fin_jitter_map = np.zeros(w, dtype=np.float64)
    for i in range(num_fins):
        start = max(0, int(i * fin_period - x_offset))
        end = min(w, int((i + 1) * fin_period - x_offset))
        if start < end:
            fin_jitter_map[start:end] = fin_jitter[i]

    fin_layer = fin_mask * (config.fin_intensity - config.background_intensity + fin_jitter_map)
    image += fin_layer[np.newaxis, :]

    # --- Horizontal gates ---
    gate_half = gate_period * config.gate_width_frac / 2.0
    y_coords = np.arange(h, dtype=np.float64)
    gate_phase = np.mod(y_offset + y_coords, gate_period)
    gate_center_dist = np.abs(gate_phase - gate_period / 2.0)
    gate_mask = np.clip(1.0 - (gate_center_dist - gate_half) / 2.0, 0.0, 1.0)
    # Per-gate jitter
    num_gates = int(np.ceil((h + y_offset) / gate_period))
    gate_jitter = rng.uniform(-3.0, 3.0, num_gates)
    gate_jitter_map = np.zeros(h, dtype=np.float64)
    for i in range(num_gates):
        start = max(0, int(i * gate_period - y_offset))
        end = min(h, int((i + 1) * gate_period - y_offset))
        if start < end:
            gate_jitter_map[start:end] = gate_jitter[i]

    gate_layer = gate_mask * (config.gate_intensity - config.background_intensity + gate_jitter_map)
    image += gate_layer[:, np.newaxis]

    # --- Intersection brightening ---
    intersection = fin_mask[np.newaxis, :] * gate_mask[:, np.newaxis]
    image += intersection * config.intersection_boost

    # --- Low-frequency manufacturing variation ---
    lf_h = max(2, h // 40)
    lf_w = max(2, w // 40)
    low_freq = rng.normal(0.0, config.mfg_variation_sigma, (lf_h, lf_w))
    low_freq_upsampled = cv2.resize(
        low_freq.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
    ).astype(np.float64)
    image += low_freq_upsampled

    # --- Subtle structural texture ---
    texture = rng.uniform(-2.0, 2.0, (h, w))
    image += texture

    # --- Global defects (scattered across entire image) ---
    for _ in range(config.num_global_defects):
        dw = int(rng.integers(config.global_defect_min_size, config.global_defect_max_size + 1))
        dh = int(rng.integers(config.global_defect_min_size, config.global_defect_max_size + 1))
        dx = int(rng.integers(0, max(1, w - dw)))
        dy = int(rng.integers(0, max(1, h - dh)))
        defect_type = int(rng.integers(0, 3))
        if defect_type == 0:
            val = float(rng.uniform(160.0, 230.0))
            image[dy:dy + dh, dx:dx + dw] = val
        elif defect_type == 1:
            val = float(rng.uniform(20.0, 60.0))
            image[dy:dy + dh, dx:dx + dw] = val
        else:
            # Thin scratch
            if rng.random() < 0.5:
                image[dy, dx:dx + dw] = float(rng.uniform(40.0, 220.0))
            else:
                image[dy:dy + dh, dx] = float(rng.uniform(40.0, 220.0))

    return np.clip(image, 0.0, 255.0)


def _inject_global_defects(
    image: np.ndarray,
    config: GeneratorConfig,
    rng: Generator,
) -> np.ndarray:
    """Inject global (background) defects scattered across the entire image."""
    modified = image.copy()
    h, w = image.shape[:2]
    for _ in range(config.num_global_defects):
        dw = int(rng.integers(config.global_defect_min_size, config.global_defect_max_size + 1))
        dh = int(rng.integers(config.global_defect_min_size, config.global_defect_max_size + 1))
        dx = int(rng.integers(0, max(1, w - dw)))
        dy = int(rng.integers(0, max(1, h - dh)))
        defect_type = int(rng.integers(0, 3))
        if defect_type == 0:
            val = float(rng.uniform(160.0, 230.0))
            modified[dy:dy + dh, dx:dx + dw] = val
        elif defect_type == 1:
            val = float(rng.uniform(20.0, 60.0))
            modified[dy:dy + dh, dx:dx + dw] = val
        else:
            if rng.random() < 0.5:
                modified[dy, dx:dx + dw] = float(rng.uniform(40.0, 220.0))
            else:
                modified[dy:dy + dh, dx] = float(rng.uniform(40.0, 220.0))
    return np.clip(modified, 0.0, 255.0)


def _inject_target_defects(
    image: np.ndarray,
    target_y: int,
    target_x: int,
    target_h: int,
    target_w: int,
    config: GeneratorConfig,
    rng: Generator,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Inject realistic localized process defects into the target region.
    The defect parameters are generated in the "Search" (physical) coordinate space,
    but applied to the `image` scaled by `scale`.

    Args:
        image: The base image to modify.
        target_y: Top-left y of target region in image coordinates.
        target_x: Top-left x of target region in image coordinates.
        target_h: Height of target region (in image coordinates).
        target_w: Width of target region (in image coordinates).
        config: Generator configuration.
        rng: Deterministic random generator.
        scale: The scale factor (e.g. ~10.0 for Reference, 1.0 for Search).

    Returns:
        Modified image with defects injected.
    """
    modified = image.copy()
    img_h, img_w = image.shape[:2]

    # The physical target dimensions (in Search space)
    phys_w = int(round(target_w / scale))
    phys_h = int(round(target_h / scale))

    num_defects = config.num_target_defects

    for _ in range(num_defects):
        defect_type = int(rng.integers(0, 7))

        if defect_type == 0:
            phys_gap_w = int(rng.integers(max(2, config.defect_min_size), min(config.defect_max_size, max(3, phys_w // 3)) + 1))
            phys_gap_h = int(rng.integers(max(4, config.defect_min_size), min(config.defect_max_size * 2, max(5, phys_h // 2)) + 1))
            phys_local_x = int(rng.integers(2, max(3, phys_w - phys_gap_w - 2)))
            phys_local_y = int(rng.integers(2, max(3, phys_h - phys_gap_h - 2)))
            
            abs_y = target_y + int(round(phys_local_y * scale))
            abs_x = target_x + int(round(phys_local_x * scale))
            gap_h = int(round(phys_gap_h * scale))
            gap_w = int(round(phys_gap_w * scale))
            y_end = min(img_h, abs_y + gap_h)
            x_end = min(img_w, abs_x + gap_w)
            val = config.background_intensity + float(rng.uniform(-8.0, 8.0))
            modified[abs_y:y_end, abs_x:x_end] = val

        elif defect_type == 1:
            phys_gap_w = int(rng.integers(max(6, config.defect_min_size), min(config.defect_max_size * 2, max(7, phys_w // 2)) + 1))
            phys_gap_h = int(rng.integers(max(2, config.defect_min_size), min(config.defect_max_size, max(3, phys_h // 4)) + 1))
            phys_local_x = int(rng.integers(2, max(3, phys_w - phys_gap_w - 2)))
            phys_local_y = int(rng.integers(2, max(3, phys_h - phys_gap_h - 2)))
            
            abs_y = target_y + int(round(phys_local_y * scale))
            abs_x = target_x + int(round(phys_local_x * scale))
            gap_h = int(round(phys_gap_h * scale))
            gap_w = int(round(phys_gap_w * scale))
            y_end = min(img_h, abs_y + gap_h)
            x_end = min(img_w, abs_x + gap_w)
            val = config.background_intensity + float(rng.uniform(-5.0, 5.0))
            modified[abs_y:y_end, abs_x:x_end] = val

        elif defect_type == 2:
            phys_patch_w = int(rng.integers(config.defect_min_size, min(config.defect_max_size, max(4, phys_w // 3)) + 1))
            phys_patch_h = int(rng.integers(config.defect_min_size, min(config.defect_max_size, max(4, phys_h // 3)) + 1))
            phys_local_x = int(rng.integers(1, max(2, phys_w - phys_patch_w - 1)))
            phys_local_y = int(rng.integers(1, max(2, phys_h - phys_patch_h - 1)))
            
            abs_y = target_y + int(round(phys_local_y * scale))
            abs_x = target_x + int(round(phys_local_x * scale))
            patch_h = int(round(phys_patch_h * scale))
            patch_w = int(round(phys_patch_w * scale))
            y_end = min(img_h, abs_y + patch_h)
            x_end = min(img_w, abs_x + patch_w)
            shift = float(rng.uniform(
                config.defect_intensity_low - config.background_intensity,
                config.defect_intensity_high - config.fin_intensity,
            ))
            modified[abs_y:y_end, abs_x:x_end] += shift

        elif defect_type == 3:
            num_spots = int(rng.integers(4, 12))
            for _ in range(num_spots):
                phys_spot_y = int(rng.integers(1, max(2, phys_h - 1)))
                phys_spot_x = int(rng.integers(1, max(2, phys_w - 1)))
                phys_radius = int(rng.integers(1, 4))
                
                spot_y = target_y + int(round(phys_spot_y * scale))
                spot_x = target_x + int(round(phys_spot_x * scale))
                radius = int(round(phys_radius * scale))
                
                y_lo = max(0, spot_y - radius)
                y_hi = min(img_h, spot_y + radius + 1)
                x_lo = max(0, spot_x - radius)
                x_hi = min(img_w, spot_x + radius + 1)
                if rng.random() > 0.5:
                    modified[y_lo:y_hi, x_lo:x_hi] = float(rng.uniform(180.0, 240.0))
                else:
                    modified[y_lo:y_hi, x_lo:x_hi] = float(rng.uniform(10.0, 50.0))

        elif defect_type == 4:
            num_bumps = int(rng.integers(5, 15))
            for _ in range(num_bumps):
                phys_bump_y = int(rng.integers(0, phys_h))
                phys_bump_x = int(rng.integers(0, phys_w))
                phys_bh = int(rng.integers(1, 5))
                phys_bw = int(rng.integers(1, 5))
                
                bump_y = target_y + int(round(phys_bump_y * scale))
                bump_x = target_x + int(round(phys_bump_x * scale))
                bh = int(round(phys_bh * scale))
                bw = int(round(phys_bw * scale))
                
                y_lo = max(0, bump_y)
                y_hi = min(img_h, bump_y + bh)
                x_lo = max(0, bump_x)
                x_hi = min(img_w, bump_x + bw)
                shift = float(rng.uniform(-50.0, 50.0))
                modified[y_lo:y_hi, x_lo:x_hi] += shift

        elif defect_type == 5:
            phys_bridge_w = int(rng.integers(max(3, config.defect_min_size), min(config.defect_max_size, max(4, phys_w // 3)) + 1))
            phys_bridge_h = int(rng.integers(2, min(6, max(3, phys_h // 4)) + 1))
            phys_local_x = int(rng.integers(2, max(3, phys_w - phys_bridge_w - 2)))
            phys_local_y = int(rng.integers(2, max(3, phys_h - phys_bridge_h - 2)))
            
            abs_y = target_y + int(round(phys_local_y * scale))
            abs_x = target_x + int(round(phys_local_x * scale))
            bridge_h = int(round(phys_bridge_h * scale))
            bridge_w = int(round(phys_bridge_w * scale))
            y_end = min(img_h, abs_y + bridge_h)
            x_end = min(img_w, abs_x + bridge_w)
            val = float(rng.uniform(config.fin_intensity, config.fin_intensity + 40.0))
            modified[abs_y:y_end, abs_x:x_end] = val

        else:
            phys_cx = int(rng.integers(4, max(5, phys_w - 4)))
            phys_cy = int(rng.integers(4, max(5, phys_h - 4)))
            phys_radius = int(rng.integers(2, min(8, max(3, min(phys_w, phys_h) // 6)) + 1))
            
            cx = target_x + int(round(phys_cx * scale))
            cy = target_y + int(round(phys_cy * scale))
            radius = int(round(phys_radius * scale))
            
            yy, xx = np.ogrid[
                max(0, cy - radius):min(img_h, cy + radius + 1),
                max(0, cx - radius):min(img_w, cx + radius + 1),
            ]
            cy_off = cy - max(0, cy - radius)
            cx_off = cx - max(0, cx - radius)
            dist_sq = (np.arange(yy.shape[0])[:, np.newaxis] - cy_off) ** 2 + \
                       (np.arange(xx.shape[1])[np.newaxis, :] - cx_off) ** 2
            mask = dist_sq <= radius ** 2
            region = modified[
                max(0, cy - radius):min(img_h, cy + radius + 1),
                max(0, cx - radius):min(img_w, cx + radius + 1),
            ]
            if rng.random() > 0.5:
                region[mask] = float(rng.uniform(200.0, 250.0))
            else:
                region[mask] = float(rng.uniform(5.0, 40.0))

    return np.clip(modified, 0.0, 255.0)


def _add_illumination_gradient(
    image: np.ndarray,
    config: GeneratorConfig,
    rng: Generator,
) -> np.ndarray:
    """
    Add a smooth illumination gradient to simulate non-uniform SEM lighting.

    Args:
        image: Input image (float64).
        config: Generator configuration.
        rng: Random generator.

    Returns:
        Image with illumination gradient applied, clipped to [0, 255].
    """
    h, w = image.shape[:2]
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    magnitude = float(rng.uniform(0.0, config.illumination_gradient_max))

    y_norm = np.linspace(-1.0, 1.0, h)[:, np.newaxis]
    x_norm = np.linspace(-1.0, 1.0, w)[np.newaxis, :]

    gradient = magnitude * (np.cos(angle) * x_norm + np.sin(angle) * y_norm)
    return np.clip(image + gradient, 0.0, 255.0)


def _apply_edge_brightening(
    image: np.ndarray,
    config: GeneratorConfig,
    rng: Generator,
) -> np.ndarray:
    """
    Boost intensity along feature edges to mimic SEM secondary-electron edge contrast.
    """
    # Use Sobel for edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_mask = np.hypot(sobel_x, sobel_y)
    
    # Normalize edge mask roughly to 0-1 range
    edge_mask = edge_mask / (np.max(edge_mask) + 1e-5)
    
    # Blend additively
    boost = edge_mask * (255.0 * config.edge_boost_factor)
    return np.clip(image + boost, 0.0, 255.0)


def _apply_sensor_noise(
    image: np.ndarray,
    noise_sigma: float,
    speckle_sigma: float,
    rng: Generator,
) -> np.ndarray:
    """
    Apply independent Speckle and Gaussian sensor noise.

    Args:
        image: Clean image (float64).
        noise_sigma: Standard deviation of additive Gaussian noise.
        speckle_sigma: Standard deviation of multiplicative Speckle noise.
        rng: Random generator.

    Returns:
        Noisy image as uint8.
    """
    # Multiplicative Speckle Noise
    speckle_factor = rng.normal(0.0, speckle_sigma, image.shape)
    noisy = image * (1.0 + speckle_factor)
    
    # Additive Gaussian Noise
    noisy = noisy + rng.normal(0.0, noise_sigma, image.shape)
    
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def generate_sample(
    seed: int,
    config: Optional[GeneratorConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, SampleMetadata]:
    """
    Generate a single synthetic FinFET sample pair.
    """
    if config is None:
        config = GeneratorConfig()

    rng = np.random.default_rng(seed)

    s_h, s_w = config.search_size, config.search_size
    footprint_size = config.ref_size

    # --- Phase B.c: CRITICAL MARGIN CHECK ---
    # Ensure the footprint spans at least 1.5x the native fin period
    min_required_footprint = int(np.ceil(1.5 * config.fin_period))
    if footprint_size < min_required_footprint:
        footprint_size = min_required_footprint

    # --- 1. Sample random scale ---
    scale = float(rng.uniform(config.scale_min, config.scale_max))
    
    # --- 2. Validate geometry ---
    if footprint_size >= s_w or footprint_size >= s_h:
        raise InvalidSampleError(
            f"Footprint {footprint_size}x{footprint_size} does not fit in "
            f"search image {s_w}x{s_h}."
        )

    margin = config.placement_margin
    max_x = s_w - footprint_size - margin
    max_y = s_h - footprint_size - margin
    if max_x < margin or max_y < margin:
        raise InvalidSampleError(
            f"Not enough room to place footprint ({footprint_size}x{footprint_size}) "
            f"with margin={margin} in search image ({s_w}x{s_h})."
        )

    # --- Phase B.a: Render Search base ---
    search_clean = _generate_finfet_base(
        s_h, s_w, rng, config,
        x_offset=0.0, y_offset=0.0,
        fin_period=config.fin_period,
        gate_period=config.gate_period,
    )

    # --- Phase B.b: Choose placement ---
    gt_x = int(rng.integers(margin, max_x + 1))
    gt_y = int(rng.integers(margin, max_y + 1))

    # --- Phase B.d: Render Reference SEPARATELY ---
    # Note: user instruction says `fin_period_fine = fin_period / scale`, but ratio verification
    # expects reference_period to be larger (e.g. 140 vs 14), so it must be fin_period * scale.
    fin_period_ref = config.fin_period * scale
    gate_period_ref = config.gate_period * scale
    x_offset_ref = gt_x * scale
    y_offset_ref = gt_y * scale
    
    ref_h = int(round(footprint_size * scale))
    ref_w = int(round(footprint_size * scale))

    ref_clean = _generate_finfet_base(
        ref_h, ref_w, rng, config,
        x_offset=x_offset_ref, y_offset=y_offset_ref,
        fin_period=fin_period_ref,
        gate_period=gate_period_ref,
    )

    # --- Phase B.e: Inject defects into BOTH renderings at the same location ---
    rng_state = rng.bit_generator.state
    
    # Inject into Search footprint (scale=1.0)
    search_defected = _inject_target_defects(
        search_clean, gt_y, gt_x, footprint_size, footprint_size, config, rng, scale=1.0
    )
    
    # Restore RNG state and inject into Reference (scale=scale)
    rng.bit_generator.state = rng_state
    ref_defected = _inject_target_defects(
        ref_clean, 0, 0, ref_h, ref_w, config, rng, scale=scale
    )

    # --- Apply global defects to Search (outside footprint) ---
    search_defected = _inject_global_defects(search_defected, config, rng)

    # --- Apply illumination gradient ---
    illuminated_search = _add_illumination_gradient(search_defected, config, rng)

    # --- Apply Edge Brightening (SEM mimic) ---
    search_edge = _apply_edge_brightening(illuminated_search, config, rng)
    ref_edge = _apply_edge_brightening(ref_defected, config, rng)

    # --- Apply Blur ---
    search_blur_sigma = float(rng.uniform(config.blur_sigma_search_min, config.blur_sigma_search_max))
    ref_blur_sigma = float(rng.uniform(config.blur_sigma_ref_min, config.blur_sigma_ref_max))
    search_blurred = cv2.GaussianBlur(search_edge, (5, 5), search_blur_sigma)
    if ref_blur_sigma > 0:
        ref_blurred = cv2.GaussianBlur(ref_edge, (5, 5), ref_blur_sigma)
    else:
        ref_blurred = ref_edge

    # --- Apply sensor noise ---
    search_img = _apply_sensor_noise(search_blurred, config.noise_sigma_search, config.speckle_sigma, rng)

    b_offset = float(rng.uniform(-config.brightness_jitter, config.brightness_jitter))
    c_factor = 1.0 + float(rng.uniform(-config.contrast_jitter, config.contrast_jitter))
    ref_jittered = ref_blurred * c_factor + b_offset
    ref_img = _apply_sensor_noise(ref_jittered, config.noise_sigma_ref, config.speckle_sigma, rng)

    # --- Apply Rotation to Reference ---
    # Randomly rotate reference image to simulate angular drift
    angle = float(rng.uniform(-config.rotation_range, config.rotation_range))
    center = (ref_img.shape[1] / 2.0, ref_img.shape[0] / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    ref_img = cv2.warpAffine(
        ref_img, 
        rot_mat, 
        (ref_img.shape[1], ref_img.shape[0]), 
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # --- Quality checks ---
    ref_var = float(np.var(ref_img.astype(np.float64)))
    search_var = float(np.var(search_img.astype(np.float64)))
    if ref_var < config.min_variance:
        raise InvalidSampleError(f"Reference variance too low: {ref_var:.2f}")
    if search_var < config.min_variance:
        raise InvalidSampleError(f"Search variance too low: {search_var:.2f}")

    gt_center_x = float(gt_x) + (footprint_size - 1) / 2.0
    gt_center_y = float(gt_y) + (footprint_size - 1) / 2.0

    metadata = SampleMetadata(
        gt_top_left_x=gt_x,
        gt_top_left_y=gt_y,
        gt_center_x=gt_center_x,
        gt_center_y=gt_center_y,
        scaled_width=footprint_size,
        scaled_height=footprint_size,
        scale=round(float(scale), 6),
        seed=seed,
        ref_width=ref_w,
        ref_height=ref_h,
        config=_config_to_dict(config),
    )

    return ref_img, search_img, metadata


# ---------------------------------------------------------------------------
# Batch / Dataset Generation
# ---------------------------------------------------------------------------

def generate_batch(
    num_samples: int,
    config: Optional[GeneratorConfig] = None,
    base_seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, SampleMetadata]]:
    """
    Generate a batch of dataset samples in memory.

    Each sample uses seed = base_seed + index.

    Args:
        num_samples: Number of samples to generate.
        config: Generator configuration. Uses defaults if None.
        base_seed: Starting seed value.

    Returns:
        List of (ref_img, search_img, metadata) tuples.
    """
    if config is None:
        config = GeneratorConfig()

    samples: List[Tuple[np.ndarray, np.ndarray, SampleMetadata]] = []
    for i in range(num_samples):
        samples.append(generate_sample(seed=base_seed + i, config=config))
    return samples


def generate_dataset(
    output_directory: str,
    number_of_samples: int,
    start_seed: int = 0,
    config: Optional[GeneratorConfig] = None,
    split: str = "train",
) -> List[SampleMetadata]:
    """
    Generate a full dataset of synthetic FinFET samples and write to disk.

    Directory layout::

        <output_directory>/
            <split>/
                sample_000/
                    reference.png
                    search.png
                    ground_truth.json
                sample_001/
                    ...
            metadata_<split>.json

    Args:
        output_directory: Root directory for the dataset.
        number_of_samples: Number of samples to generate.
        start_seed: Starting RNG seed (incremented per sample).
        config: Generator config; uses defaults if None.
        split: Subdirectory name (e.g. "train", "validation").

    Returns:
        List of SampleMetadata for all successfully generated samples.
    """
    if config is None:
        config = GeneratorConfig()

    split_dir = os.path.join(output_directory, split)
    os.makedirs(split_dir, exist_ok=True)

    all_meta: List[SampleMetadata] = []
    sample_idx = 0
    current_seed = start_seed
    max_attempts = number_of_samples * 3

    while sample_idx < number_of_samples and max_attempts > 0:
        max_attempts -= 1
        try:
            ref_img, search_img, meta = generate_sample(current_seed, config)
        except InvalidSampleError as exc:
            logger.warning("Seed %d rejected: %s", current_seed, exc)
            current_seed += 1
            continue

        sample_dir = os.path.join(split_dir, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        ref_path = os.path.join(sample_dir, "reference.png")
        search_path = os.path.join(sample_dir, "search.png")
        gt_path = os.path.join(sample_dir, "ground_truth.json")

        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)

        meta_dict = asdict(meta)
        with open(gt_path, "w") as f:
            json.dump(meta_dict, f, indent=4)

        all_meta.append(meta)
        sample_idx += 1
        current_seed += 1

    # Master metadata
    master: Dict[str, Any] = {
        "split": split,
        "num_samples": len(all_meta),
        "start_seed": start_seed,
        "config": _config_to_dict(config),
        "samples": [
            {
                "sample_id": f"sample_{i:03d}",
                "seed": m.seed,
                "gt_top_left_x": m.gt_top_left_x,
                "gt_top_left_y": m.gt_top_left_y,
                "gt_center_x": m.gt_center_x,
                "gt_center_y": m.gt_center_y,
                "scale": m.scale,
            }
            for i, m in enumerate(all_meta)
        ],
    }
    with open(os.path.join(output_directory, f"metadata_{split}.json"), "w") as f:
        json.dump(master, f, indent=4)

    logger.info(
        "Generated %d/%d samples in '%s'.",
        len(all_meta), number_of_samples, split_dir,
    )
    return all_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_to_dict(config: GeneratorConfig) -> Dict[str, Any]:
    """Convert a GeneratorConfig to a JSON-serializable dict."""
    return asdict(config)
