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

    Returns:
        float64 image of shape (h, w), values approximately in [0, 255].
    """
    image = np.full((h, w), config.background_intensity, dtype=np.float64)

    # --- Vertical fins ---
    fin_half = config.fin_period * config.fin_width_frac / 2.0
    x_coords = np.arange(w, dtype=np.float64)
    # Distance from nearest fin center
    fin_phase = np.mod(x_coords, config.fin_period)
    fin_center_dist = np.abs(fin_phase - config.fin_period / 2.0)
    # Smooth fin profile: bright where close to center, dark in space
    fin_mask = np.clip(1.0 - (fin_center_dist - fin_half) / 2.0, 0.0, 1.0)
    # Per-fin jitter
    num_fins = int(np.ceil(w / config.fin_period))
    fin_jitter = rng.uniform(-4.0, 4.0, num_fins)
    fin_jitter_map = np.zeros(w, dtype=np.float64)
    for i in range(num_fins):
        start = int(i * config.fin_period)
        end = min(w, int((i + 1) * config.fin_period))
        if start < end:
            fin_jitter_map[start:end] = fin_jitter[i]

    fin_layer = fin_mask * (config.fin_intensity - config.background_intensity + fin_jitter_map)
    image += fin_layer[np.newaxis, :]

    # --- Horizontal gates ---
    gate_half = config.gate_period * config.gate_width_frac / 2.0
    y_coords = np.arange(h, dtype=np.float64)
    gate_phase = np.mod(y_coords, config.gate_period)
    gate_center_dist = np.abs(gate_phase - config.gate_period / 2.0)
    gate_mask = np.clip(1.0 - (gate_center_dist - gate_half) / 2.0, 0.0, 1.0)
    # Per-gate jitter
    num_gates = int(np.ceil(h / config.gate_period))
    gate_jitter = rng.uniform(-3.0, 3.0, num_gates)
    gate_jitter_map = np.zeros(h, dtype=np.float64)
    for i in range(num_gates):
        start = int(i * config.gate_period)
        end = min(h, int((i + 1) * config.gate_period))
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


def _inject_target_defects(
    image: np.ndarray,
    target_y: int,
    target_x: int,
    target_h: int,
    target_w: int,
    config: GeneratorConfig,
    rng: Generator,
) -> np.ndarray:
    """
    Inject realistic localized process defects into the target region.

    These defects make the target region structurally distinctive from the
    surrounding periodic pattern.  They resemble real semiconductor defects:

    Type 0 — Missing fin segment: vertical fin line has a gap
    Type 1 — Gate interruption: horizontal gate line has a break
    Type 2 — Line-width variation: local thickening/thinning of structure
    Type 3 — Defect cluster: multiple small bright/dark spots
    Type 4 — Edge roughness: irregular bumps along structure edges
    Type 5 — Bridging defect: connects two adjacent fins/gates
    Type 6 — Void/particle: circular bright or dark region

    All defects are confined to the target region.

    Args:
        image: The base FinFET image (modified IN PLACE and returned).
        target_y: Top-left y of target region in image coordinates.
        target_x: Top-left x of target region in image coordinates.
        target_h: Height of target region.
        target_w: Width of target region.
        config: Generator configuration.
        rng: Deterministic random generator.

    Returns:
        Modified image with defects injected.
    """
    modified = image.copy()
    img_h, img_w = image.shape[:2]

    num_defects = config.num_target_defects

    for _ in range(num_defects):
        defect_type = int(rng.integers(0, 7))

        if defect_type == 0:
            # Missing fin segment: dark gap in a bright fin
            gap_w = int(rng.integers(max(2, config.defect_min_size),
                                      min(config.defect_max_size, target_w // 3) + 1))
            gap_h = int(rng.integers(max(4, config.defect_min_size),
                                      min(config.defect_max_size * 2, target_h // 2) + 1))
            local_x = int(rng.integers(2, max(3, target_w - gap_w - 2)))
            local_y = int(rng.integers(2, max(3, target_h - gap_h - 2)))
            abs_y = target_y + local_y
            abs_x = target_x + local_x
            y_end = min(img_h, abs_y + gap_h)
            x_end = min(img_w, abs_x + gap_w)
            val = config.background_intensity + float(rng.uniform(-8.0, 8.0))
            modified[abs_y:y_end, abs_x:x_end] = val

        elif defect_type == 1:
            # Gate interruption: dark break in a gate line
            gap_w = int(rng.integers(max(6, config.defect_min_size),
                                      min(config.defect_max_size * 2, target_w // 2) + 1))
            gap_h = int(rng.integers(max(2, config.defect_min_size),
                                      min(config.defect_max_size, target_h // 4) + 1))
            local_x = int(rng.integers(2, max(3, target_w - gap_w - 2)))
            local_y = int(rng.integers(2, max(3, target_h - gap_h - 2)))
            abs_y = target_y + local_y
            abs_x = target_x + local_x
            y_end = min(img_h, abs_y + gap_h)
            x_end = min(img_w, abs_x + gap_w)
            val = config.background_intensity + float(rng.uniform(-5.0, 5.0))
            modified[abs_y:y_end, abs_x:x_end] = val

        elif defect_type == 2:
            # Line-width variation: intensity shift in a local region
            patch_w = int(rng.integers(config.defect_min_size,
                                        min(config.defect_max_size, target_w // 3) + 1))
            patch_h = int(rng.integers(config.defect_min_size,
                                        min(config.defect_max_size, target_h // 3) + 1))
            local_x = int(rng.integers(1, max(2, target_w - patch_w - 1)))
            local_y = int(rng.integers(1, max(2, target_h - patch_h - 1)))
            abs_y = target_y + local_y
            abs_x = target_x + local_x
            y_end = min(img_h, abs_y + patch_h)
            x_end = min(img_w, abs_x + patch_w)
            shift = float(rng.uniform(
                config.defect_intensity_low - config.background_intensity,
                config.defect_intensity_high - config.fin_intensity,
            ))
            modified[abs_y:y_end, abs_x:x_end] += shift

        elif defect_type == 3:
            # Defect cluster: multiple small spots
            num_spots = int(rng.integers(4, 12))
            for _ in range(num_spots):
                spot_y = target_y + int(rng.integers(1, max(2, target_h - 1)))
                spot_x = target_x + int(rng.integers(1, max(2, target_w - 1)))
                radius = int(rng.integers(1, 4))
                y_lo = max(0, spot_y - radius)
                y_hi = min(img_h, spot_y + radius + 1)
                x_lo = max(0, spot_x - radius)
                x_hi = min(img_w, spot_x + radius + 1)
                if rng.random() > 0.5:
                    modified[y_lo:y_hi, x_lo:x_hi] = float(rng.uniform(180.0, 240.0))
                else:
                    modified[y_lo:y_hi, x_lo:x_hi] = float(rng.uniform(10.0, 50.0))

        elif defect_type == 4:
            # Edge roughness: bumps along structure edges
            num_bumps = int(rng.integers(5, 15))
            for _ in range(num_bumps):
                bump_y = target_y + int(rng.integers(0, target_h))
                bump_x = target_x + int(rng.integers(0, target_w))
                bh = int(rng.integers(1, 5))
                bw = int(rng.integers(1, 5))
                y_lo = max(0, bump_y)
                y_hi = min(img_h, bump_y + bh)
                x_lo = max(0, bump_x)
                x_hi = min(img_w, bump_x + bw)
                shift = float(rng.uniform(-50.0, 50.0))
                modified[y_lo:y_hi, x_lo:x_hi] += shift

        elif defect_type == 5:
            # Bridging defect: bright bridge connecting adjacent structures
            bridge_w = int(rng.integers(
                max(3, config.defect_min_size),
                min(config.defect_max_size, target_w // 3) + 1,
            ))
            bridge_h = int(rng.integers(2, min(6, target_h // 4) + 1))
            local_x = int(rng.integers(2, max(3, target_w - bridge_w - 2)))
            local_y = int(rng.integers(2, max(3, target_h - bridge_h - 2)))
            abs_y = target_y + local_y
            abs_x = target_x + local_x
            y_end = min(img_h, abs_y + bridge_h)
            x_end = min(img_w, abs_x + bridge_w)
            val = float(rng.uniform(config.fin_intensity, config.fin_intensity + 40.0))
            modified[abs_y:y_end, abs_x:x_end] = val

        else:
            # Void/particle: circular defect
            cx = target_x + int(rng.integers(4, max(5, target_w - 4)))
            cy = target_y + int(rng.integers(4, max(5, target_h - 4)))
            radius = int(rng.integers(2, min(8, min(target_w, target_h) // 6) + 1))
            yy, xx = np.ogrid[
                max(0, cy - radius):min(img_h, cy + radius + 1),
                max(0, cx - radius):min(img_w, cx + radius + 1),
            ]
            # Compute actual center offsets relative to ogrid start
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


def _apply_sensor_noise(
    image: np.ndarray,
    noise_sigma: float,
    rng: Generator,
) -> np.ndarray:
    """
    Apply independent Gaussian sensor noise and convert to uint8.

    Args:
        image: Clean image (float64).
        noise_sigma: Standard deviation of Gaussian noise.
        rng: Random generator.

    Returns:
        Noisy image as uint8.
    """
    noisy = image + rng.normal(0.0, noise_sigma, image.shape)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Single-Sample Generation
# ---------------------------------------------------------------------------

def generate_sample(
    seed: int,
    config: Optional[GeneratorConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, SampleMetadata]:
    """
    Generate a single synthetic FinFET sample pair.

    Pipeline:
        1. Sample random scale within configured jitter range
        2. Validate geometry (scaled template fits in search image)
        3. Generate clean FinFET base structure
        4. Choose random placement with margin
        5. Inject realistic process defects into the target region
        6. Apply illumination gradient
        7. Create search image with independent sensor noise
        8. Extract reference region at original scale with independent noise
        9. Compute ground-truth coordinates (both top-left and center)

    Args:
        seed: Deterministic RNG seed.
        config: Generator configuration.  Uses defaults if None.

    Returns:
        (ref_img, search_img, metadata)
            ref_img:    uint8, shape (ref_size, ref_size)
            search_img: uint8, shape (search_size, search_size)
            metadata:   SampleMetadata with ground truth

    Raises:
        InvalidSampleError: If the geometry is invalid or quality checks fail.
    """
    if config is None:
        config = GeneratorConfig()

    rng = np.random.default_rng(seed)

    s_h, s_w = config.search_size, config.search_size
    r_h, r_w = config.ref_size, config.ref_size

    # --- 1. Sample random scale ---
    scale = float(rng.uniform(config.scale_min, config.scale_max))
    target_w = int(round(r_w * scale))
    target_h = int(round(r_h * scale))

    # --- 2. Validate geometry ---
    if target_w >= s_w or target_h >= s_h:
        raise InvalidSampleError(
            f"Scaled patch {target_w}×{target_h} does not fit in "
            f"search image {s_w}×{s_h}."
        )

    margin = config.placement_margin
    max_x = s_w - target_w - margin
    max_y = s_h - target_h - margin
    if max_x < margin or max_y < margin:
        raise InvalidSampleError(
            f"Not enough room to place scaled template ({target_w}×{target_h}) "
            f"with margin={margin} in search image ({s_w}×{s_h})."
        )

    # --- 3. Generate clean FinFET base ---
    base_layout = _generate_finfet_base(s_h, s_w, rng, config)

    # --- 4. Choose random placement ---
    gt_x = int(rng.integers(margin, max_x + 1))
    gt_y = int(rng.integers(margin, max_y + 1))

    # --- 5. Inject target-specific defects ---
    defected_layout = _inject_target_defects(
        base_layout, gt_y, gt_x, target_h, target_w, config, rng,
    )

    # --- 6. Apply illumination gradient ---
    illuminated = _add_illumination_gradient(defected_layout, config, rng)

    # --- 7. Create search image with independent noise ---
    search_img = _apply_sensor_noise(illuminated, config.noise_sigma_search, rng)

    # --- 8. Extract reference from clean-with-defects at original scale ---
    # Use the defected (but pre-noise) structure to extract reference.
    # This ensures reference has the defect structure but NOT the search noise.
    target_region_clean = defected_layout[
        gt_y:gt_y + target_h,
        gt_x:gt_x + target_w,
    ].copy()

    # Downscale to reference size
    ref_clean = cv2.resize(
        target_region_clean.astype(np.float32),
        (r_w, r_h),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)

    # Apply brightness/contrast jitter
    b_offset = float(rng.uniform(-config.brightness_jitter, config.brightness_jitter))
    c_factor = 1.0 + float(rng.uniform(-config.contrast_jitter, config.contrast_jitter))
    ref_clean = ref_clean * c_factor + b_offset

    # Apply independent reference noise
    ref_img = _apply_sensor_noise(ref_clean, config.noise_sigma_ref, rng)

    # --- Quality checks ---
    ref_var = float(np.var(ref_img.astype(np.float64)))
    search_var = float(np.var(search_img.astype(np.float64)))
    if ref_var < config.min_variance:
        raise InvalidSampleError(f"Reference variance too low: {ref_var:.2f}")
    if search_var < config.min_variance:
        raise InvalidSampleError(f"Search variance too low: {search_var:.2f}")

    if ref_img.shape != (r_h, r_w):
        raise InvalidSampleError(f"Reference shape mismatch: {ref_img.shape}")
    if search_img.shape != (s_h, s_w):
        raise InvalidSampleError(f"Search shape mismatch: {search_img.shape}")

    # --- 9. Ground truth ---
    gt_center_x = float(gt_x) + (target_w - 1) / 2.0
    gt_center_y = float(gt_y) + (target_h - 1) / 2.0

    metadata = SampleMetadata(
        gt_top_left_x=gt_x,
        gt_top_left_y=gt_y,
        gt_center_x=gt_center_x,
        gt_center_y=gt_center_y,
        scaled_width=target_w,
        scaled_height=target_h,
        scale=round(float(scale), 6),
        seed=seed,
        ref_width=r_w,
        ref_height=r_h,
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
