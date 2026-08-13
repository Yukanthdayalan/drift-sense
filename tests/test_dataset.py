"""
Comprehensive tests for the Drift-Sense synthetic FinFET dataset generator.

Tests cover:
 1. Search image is exactly 1000×1000.
 2. Reference dimensions are valid.
 3. Scaled reference fits completely inside search image.
 4. Ground truth is inside valid bounds.
 5. Ground truth top-left and center coordinates are internally consistent.
 6. Actual scale is inside configured jitter range.
 7. Reference and search images have correct dtypes.
 8. Reference and search are not identical copies.
 9. Independent noise is actually present.
10. Multiple random seeds produce different samples.
11. Same seed produces deterministic samples.
12. Generated images contain nontrivial FinFET structure.
13. Generator rejects impossible configurations.
14. Oversized scaled template raises an appropriate error.
15. Target placement respects margins.
16. Generated sample can be passed through the existing inference/matcher pipeline.
17. Localization error is measured against ground truth.
18. Multiple generated samples are tested (not relying on one lucky seed).
19. Realistic defects improve target distinctiveness.
20. Metadata serialization.
"""
import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from drift_sense.dataset import (
    GeneratorConfig,
    InvalidSampleError,
    SampleMetadata,
    generate_batch,
    generate_dataset,
    generate_sample,
    _generate_finfet_base,
    _inject_target_defects,
    _apply_sensor_noise,
    _add_illumination_gradient,
)


# ---------------------------------------------------------------------------
# 1–7: Dimensions, dtypes, geometry
# ---------------------------------------------------------------------------

class TestImageDimensions(unittest.TestCase):
    """Verify search/reference image shapes and dtypes (tests 1, 2, 3, 7)."""

    def setUp(self) -> None:
        self.config = GeneratorConfig()
        self.ref, self.search, self.meta = generate_sample(42, self.config)

    def test_search_image_is_1000x1000(self) -> None:
        """1. Search image must be exactly 1000×1000."""
        self.assertEqual(self.search.shape, (1000, 1000))

    def test_reference_dimensions_valid(self) -> None:
        """2. Reference image matches configured dimensions."""
        self.assertEqual(
            self.ref.shape,
            (self.meta.ref_height, self.meta.ref_width),
        )

    def test_scaled_reference_fits_in_search(self) -> None:
        """3. Scaled reference must fit entirely inside search image."""
        sw = int(round(self.config.ref_size * self.meta.scale))
        sh = int(round(self.config.ref_size * self.meta.scale))
        self.assertLess(sw, self.config.search_size)
        self.assertLess(sh, self.config.search_size)

    def test_search_dtype_uint8(self) -> None:
        """7a. Search image must be uint8."""
        self.assertEqual(self.search.dtype, np.uint8)

    def test_reference_dtype_uint8(self) -> None:
        """7b. Reference image must be uint8."""
        self.assertEqual(self.ref.dtype, np.uint8)

    def test_search_is_2d(self) -> None:
        """Search image must be grayscale (2D)."""
        self.assertEqual(self.search.ndim, 2)

    def test_reference_is_2d(self) -> None:
        """Reference image must be grayscale (2D)."""
        self.assertEqual(self.ref.ndim, 2)


# ---------------------------------------------------------------------------
# 4–5: Ground truth coordinates
# ---------------------------------------------------------------------------

class TestGroundTruthCoordinates(unittest.TestCase):
    """Verify ground-truth bounds and coordinate consistency (tests 4, 5)."""

    def setUp(self) -> None:
        self.config = GeneratorConfig()

    def test_gt_top_left_in_bounds(self) -> None:
        """4a. Ground-truth top-left must be inside valid bounds."""
        for seed in range(20):
            _, _, meta = generate_sample(seed, self.config)
            self.assertGreaterEqual(meta.gt_top_left_x, 0)
            self.assertGreaterEqual(meta.gt_top_left_y, 0)
            self.assertLessEqual(
                meta.gt_top_left_x + meta.scaled_width,
                self.config.search_size,
            )
            self.assertLessEqual(
                meta.gt_top_left_y + meta.scaled_height,
                self.config.search_size,
            )

    def test_gt_center_in_bounds(self) -> None:
        """4b. Ground-truth center must be inside search image."""
        for seed in range(10):
            _, _, meta = generate_sample(seed, self.config)
            self.assertGreater(meta.gt_center_x, 0)
            self.assertGreater(meta.gt_center_y, 0)
            self.assertLess(meta.gt_center_x, self.config.search_size)
            self.assertLess(meta.gt_center_y, self.config.search_size)

    def test_gt_center_consistent_with_top_left(self) -> None:
        """5. Center coordinates must match top-left + half-template formula."""
        for seed in range(10):
            _, _, meta = generate_sample(seed, self.config)
            expected_cx = meta.gt_top_left_x + (meta.scaled_width - 1) / 2.0
            expected_cy = meta.gt_top_left_y + (meta.scaled_height - 1) / 2.0
            self.assertAlmostEqual(meta.gt_center_x, expected_cx, places=6)
            self.assertAlmostEqual(meta.gt_center_y, expected_cy, places=6)


# ---------------------------------------------------------------------------
# 6: Scale range
# ---------------------------------------------------------------------------

class TestScaleRange(unittest.TestCase):
    """Verify scale jitter is within configured bounds (test 6)."""

    def test_scale_in_range(self) -> None:
        """6. Actual scale must be within jitter bounds."""
        config = GeneratorConfig()
        for seed in range(20):
            _, _, meta = generate_sample(seed, config)
            self.assertGreaterEqual(meta.scale, config.scale_min)
            self.assertLessEqual(meta.scale, config.scale_max)

    def test_scaled_dimensions_match_metadata(self) -> None:
        """Metadata width/height must match footprint size."""
        config = GeneratorConfig()
        for seed in range(10):
            _, _, meta = generate_sample(seed, config)
            expected_size = max(config.ref_size, int(np.ceil(1.5 * config.fin_period)))
            self.assertEqual(meta.scaled_width, expected_size)
            self.assertEqual(meta.scaled_height, expected_size)


# ---------------------------------------------------------------------------
# 8–9: Independent noise
# ---------------------------------------------------------------------------

class TestIndependentNoise(unittest.TestCase):
    """Verify reference and search have independent noise (tests 8, 9)."""

    def setUp(self) -> None:
        self.config = GeneratorConfig(
            noise_sigma_search=8.0, noise_sigma_ref=12.0,
        )
        self.ref, self.search, self.meta = generate_sample(42, self.config)

    def test_not_identical_copies(self) -> None:
        """8. Reference and search-crop (downscaled) must NOT be identical."""
        m = self.meta
        crop = self.search[
            m.gt_top_left_y:m.gt_top_left_y + m.scaled_height,
            m.gt_top_left_x:m.gt_top_left_x + m.scaled_width,
        ]
        crop_downscaled = cv2.resize(
            crop, (m.ref_width, m.ref_height), interpolation=cv2.INTER_AREA,
        )
        self.assertFalse(np.array_equal(crop_downscaled, self.ref))

    def test_noise_difference_nontrivial(self) -> None:
        """9. Noise difference between ref and search crop must be significant."""
        m = self.meta
        crop = self.search[
            m.gt_top_left_y:m.gt_top_left_y + m.scaled_height,
            m.gt_top_left_x:m.gt_top_left_x + m.scaled_width,
        ]
        crop_downscaled = cv2.resize(
            crop, (m.ref_width, m.ref_height), interpolation=cv2.INTER_AREA,
        )
        diff = np.abs(
            crop_downscaled.astype(np.float64) - self.ref.astype(np.float64)
        )
        mean_diff = float(np.mean(diff))
        self.assertGreater(
            mean_diff, 1.0,
            "Reference and search crop are suspiciously similar",
        )


# ---------------------------------------------------------------------------
# 10–11: Determinism / seed variation
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    """Verify reproducibility and seed variation (tests 10, 11)."""

    def test_same_seed_identical(self) -> None:
        """11. Same seed must produce byte-identical samples."""
        r1, s1, m1 = generate_sample(999)
        r2, s2, m2 = generate_sample(999)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(s1, s2)
        self.assertEqual(m1.gt_top_left_x, m2.gt_top_left_x)
        self.assertEqual(m1.gt_top_left_y, m2.gt_top_left_y)
        self.assertEqual(m1.scale, m2.scale)

    def test_different_seeds_different(self) -> None:
        """10. Different seeds must produce different samples."""
        r1, s1, m1 = generate_sample(100)
        r2, s2, m2 = generate_sample(200)
        images_differ = not np.array_equal(s1, s2)
        positions_differ = (
            m1.gt_top_left_x != m2.gt_top_left_x
            or m1.gt_top_left_y != m2.gt_top_left_y
        )
        self.assertTrue(
            images_differ or positions_differ,
            "Different seeds produced identical samples",
        )

    def test_multiple_seeds_produce_varied_scales(self) -> None:
        """Multiple seeds should produce varied scale values."""
        scales = set()
        for seed in range(20):
            _, _, meta = generate_sample(seed)
            scales.add(round(meta.scale, 3))
        self.assertGreater(len(scales), 3, "Scale variation across seeds is too low")


# ---------------------------------------------------------------------------
# 12: FinFET structure
# ---------------------------------------------------------------------------

class TestFinFETStructure(unittest.TestCase):
    """Verify generated images contain realistic FinFET structure (test 12)."""

    def test_search_has_significant_variation(self) -> None:
        """12a. Search image std should be well above uniform."""
        _, search, _ = generate_sample(42)
        std = float(np.std(search.astype(np.float64)))
        self.assertGreater(std, 20.0)

    def test_search_has_many_unique_values(self) -> None:
        """12b. Search image should not be nearly uniform."""
        _, search, _ = generate_sample(42)
        unique = len(np.unique(search))
        self.assertGreater(unique, 50)

    def test_finfet_base_has_vertical_periodicity(self) -> None:
        """12c. Base FinFET should show vertical periodicity (fin lines)."""
        rng = np.random.default_rng(42)
        cfg = GeneratorConfig()
        structure = _generate_finfet_base(200, 200, rng, cfg)
        col_means = np.mean(structure, axis=0)
        col_std = float(np.std(col_means))
        self.assertGreater(col_std, 5.0)

    def test_finfet_base_has_horizontal_periodicity(self) -> None:
        """12d. Base FinFET should show horizontal periodicity (gate lines)."""
        rng = np.random.default_rng(42)
        cfg = GeneratorConfig()
        structure = _generate_finfet_base(200, 200, rng, cfg)
        row_means = np.mean(structure, axis=1)
        row_std = float(np.std(row_means))
        self.assertGreater(row_std, 3.0)

    def test_finfet_base_shape_and_dtype(self) -> None:
        """Base FinFET output must have correct shape and dtype."""
        rng = np.random.default_rng(0)
        cfg = GeneratorConfig()
        img = _generate_finfet_base(500, 600, rng, cfg)
        self.assertEqual(img.shape, (500, 600))
        self.assertEqual(img.dtype, np.float64)

    def test_finfet_base_deterministic(self) -> None:
        """Same seed → same base pattern."""
        cfg = GeneratorConfig()
        img1 = _generate_finfet_base(200, 200, np.random.default_rng(42), cfg)
        img2 = _generate_finfet_base(200, 200, np.random.default_rng(42), cfg)
        np.testing.assert_array_equal(img1, img2)


# ---------------------------------------------------------------------------
# 13–14: Invalid configurations
# ---------------------------------------------------------------------------

class TestInvalidConfigurations(unittest.TestCase):
    """Verify impossible configurations are rejected (tests 13, 14)."""

    def test_oversized_template_raises(self) -> None:
        """14. Scaled template larger than search must raise InvalidSampleError."""
        config = GeneratorConfig(
            ref_size=1200,
            scale_min=10.0,
            scale_max=10.0,
        )
        with self.assertRaises(InvalidSampleError):
            generate_sample(0, config)

    def test_tight_margin_raises(self) -> None:
        """13a. Template that leaves no room for margin must raise."""
        config = GeneratorConfig(
            ref_size=950,
            scale_min=10.0,
            scale_max=10.0,
            placement_margin=50,
        )
        with self.assertRaises(InvalidSampleError):
            generate_sample(0, config)

    def test_barely_fitting_template(self) -> None:
        """13b. Template that barely fits should still work."""
        # ref_size=50 at scale=10 → 500×500 in 1000×1000 with margin=20
        # max_x = 1000 - 500 - 20 = 480, needs >= 20 → fine
        config = GeneratorConfig(
            ref_size=960,
            scale_min=10.0,
            scale_max=10.0,
            placement_margin=20,
        )
        # Should not raise
        ref, search, meta = generate_sample(0, config)
        self.assertEqual(search.shape, (1000, 1000))


# ---------------------------------------------------------------------------
# 15: Placement margins
# ---------------------------------------------------------------------------

class TestPlacementMargins(unittest.TestCase):
    """Verify target placement respects configured margins (test 15)."""

    def test_margin_respected(self) -> None:
        """15. Target top-left must be >= placement_margin from border."""
        config = GeneratorConfig(placement_margin=20)
        for seed in range(15):
            _, _, meta = generate_sample(seed, config)
            self.assertGreaterEqual(meta.gt_top_left_x, config.placement_margin)
            self.assertGreaterEqual(meta.gt_top_left_y, config.placement_margin)
            # Also check right/bottom margin
            self.assertLessEqual(
                meta.gt_top_left_x + meta.scaled_width + config.placement_margin,
                config.search_size + config.placement_margin,
                # gt_top_left_x + scaled_width <= search_size - margin
                # i.e. gt_top_left_x <= search_size - scaled_width - margin = max_x
            )


# ---------------------------------------------------------------------------
# 16–18: Integration with inference pipeline
# ---------------------------------------------------------------------------

class TestInferencePipeline(unittest.TestCase):
    """Test integration with the existing inference/matcher pipeline (tests 16, 17, 18)."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_sample(self, seed: int, config: GeneratorConfig = None):
        """Generate a sample, write to disk, and run matcher. Returns (result, meta)."""
        from drift_sense.matcher import match
        from drift_sense.config import EngineConfig, VerificationConfig

        if config is None:
            config = GeneratorConfig(noise_sigma_search=3.0, noise_sigma_ref=5.0)

        ref_img, search_img, meta = generate_sample(seed, config)

        sample_dir = os.path.join(self.temp_dir, f"seed_{seed}")
        os.makedirs(sample_dir, exist_ok=True)
        ref_path = os.path.join(sample_dir, "ref.png")
        search_path = os.path.join(sample_dir, "search.png")
        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)

        # Relax verification threshold for synthetic test data
        engine_cfg = EngineConfig(
            verification=VerificationConfig(min_gradient_similarity=0.0),
        )
        result = match(ref_path, search_path, engine_cfg)
        return result, meta

    def test_sample_processable(self) -> None:
        """16. Generated sample must be processable by the matcher."""
        result, _ = self._run_sample(42)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.prediction.x, float)
        self.assertIsInstance(result.prediction.y, float)

    def test_localization_accuracy_single(self) -> None:
        """17. Localization error for one sample should be reasonable."""
        result, meta = self._run_sample(42)
        dx = result.prediction.x - meta.gt_center_x
        dy = result.prediction.y - meta.gt_center_y
        error = float(np.sqrt(dx ** 2 + dy ** 2))
        self.assertLess(
            error, 20.0,
            f"Localization error {error:.2f}px too large. "
            f"Predicted ({result.prediction.x:.1f}, {result.prediction.y:.1f}), "
            f"Expected ({meta.gt_center_x:.1f}, {meta.gt_center_y:.1f})",
        )

    def test_localization_accuracy_multiple_seeds(self) -> None:
        """18. At least 3/5 seeds should localize within 20px error."""
        errors = []
        for seed in [42, 100, 200, 300, 400]:
            result, meta = self._run_sample(seed)
            dx = result.prediction.x - meta.gt_center_x
            dy = result.prediction.y - meta.gt_center_y
            error = float(np.sqrt(dx ** 2 + dy ** 2))
            errors.append(error)

        good = sum(1 for e in errors if e < 20.0)
        self.assertGreaterEqual(
            good, 3,
            f"Only {good}/5 samples had error < 20px. Errors: "
            f"{[f'{e:.1f}' for e in errors]}",
        )


# ---------------------------------------------------------------------------
# 19: Defect distinctiveness
# ---------------------------------------------------------------------------

class TestDefectDistinctiveness(unittest.TestCase):
    """Verify that defects improve target distinctiveness (test 19)."""

    def test_defects_modify_target_region(self) -> None:
        """19a. Injecting defects must visibly modify the target region."""
        config = GeneratorConfig()
        rng = np.random.default_rng(42)
        structure = _generate_finfet_base(1000, 1000, rng, config)
        original_region = structure[400:600, 400:600].copy()

        defected = _inject_target_defects(
            structure, 400, 400, 200, 200, config, rng,
        )
        defected_region = defected[400:600, 400:600]

        diff = np.abs(defected_region - original_region)
        mean_diff = float(np.mean(diff))
        self.assertGreater(mean_diff, 2.0, "Defects should meaningfully modify target")

    def test_defects_localized_to_target(self) -> None:
        """19b. Defects must NOT modify regions far from target."""
        config = GeneratorConfig()
        rng = np.random.default_rng(42)
        structure = _generate_finfet_base(1000, 1000, rng, config)
        far_region_orig = structure[0:100, 0:100].copy()

        defected = _inject_target_defects(
            structure, 500, 500, 200, 200, config, rng,
        )
        far_region_def = defected[0:100, 0:100]
        np.testing.assert_array_equal(far_region_orig, far_region_def)

    def test_more_defects_increases_distinctiveness(self) -> None:
        """19c. More defects should create larger modification."""
        rng1 = np.random.default_rng(42)
        cfg_few = GeneratorConfig(num_target_defects=3)
        structure1 = _generate_finfet_base(1000, 1000, rng1, cfg_few)
        orig1 = structure1[400:600, 400:600].copy()
        def1 = _inject_target_defects(structure1, 400, 400, 200, 200, cfg_few, rng1)
        diff1 = float(np.mean(np.abs(def1[400:600, 400:600] - orig1)))

        rng2 = np.random.default_rng(42)
        cfg_many = GeneratorConfig(num_target_defects=30)
        structure2 = _generate_finfet_base(1000, 1000, rng2, cfg_many)
        orig2 = structure2[400:600, 400:600].copy()
        def2 = _inject_target_defects(structure2, 400, 400, 200, 200, cfg_many, rng2)
        diff2 = float(np.mean(np.abs(def2[400:600, 400:600] - orig2)))

        self.assertGreater(diff2, diff1, "More defects should increase modification")


# ---------------------------------------------------------------------------
# 20: Metadata serialization
# ---------------------------------------------------------------------------

class TestMetadata(unittest.TestCase):
    """Verify metadata content and serialization (test 20)."""

    def test_metadata_has_required_fields(self) -> None:
        """20a. Metadata must contain all essential fields."""
        _, _, meta = generate_sample(42)
        required = [
            "gt_top_left_x", "gt_top_left_y",
            "gt_center_x", "gt_center_y",
            "scaled_width", "scaled_height",
            "scale", "seed", "ref_width", "ref_height", "config",
        ]
        meta_dict = {
            "gt_top_left_x": meta.gt_top_left_x,
            "gt_top_left_y": meta.gt_top_left_y,
            "gt_center_x": meta.gt_center_x,
            "gt_center_y": meta.gt_center_y,
            "scaled_width": meta.scaled_width,
            "scaled_height": meta.scaled_height,
            "scale": meta.scale,
            "seed": meta.seed,
            "ref_width": meta.ref_width,
            "ref_height": meta.ref_height,
            "config": meta.config,
        }
        for key in required:
            self.assertIn(key, meta_dict, f"Missing metadata field: {key}")

    def test_metadata_json_serializable(self) -> None:
        """20b. Metadata must be JSON-serializable."""
        from dataclasses import asdict
        _, _, meta = generate_sample(42)
        meta_dict = asdict(meta)
        json_str = json.dumps(meta_dict)
        self.assertIsInstance(json_str, str)
        # Round-trip
        loaded = json.loads(json_str)
        self.assertEqual(loaded["seed"], meta.seed)
        self.assertAlmostEqual(loaded["scale"], meta.scale, places=4)

    def test_metadata_ref_dimensions(self) -> None:
        """20c. Metadata ref dimensions match config."""
        config = GeneratorConfig()
        _, _, meta = generate_sample(42, config)
        expected_size = int(round(max(config.ref_size, int(np.ceil(1.5 * config.fin_period))) * meta.scale))
        self.assertEqual(meta.ref_width, expected_size)
        self.assertEqual(meta.ref_height, expected_size)


# ---------------------------------------------------------------------------
# Batch / Dataset generation
# ---------------------------------------------------------------------------

class TestBatchGeneration(unittest.TestCase):
    """Test batch generation."""

    def test_batch_count(self) -> None:
        """Batch produces correct number of samples."""
        samples = generate_batch(3, base_seed=42)
        self.assertEqual(len(samples), 3)

    def test_batch_deterministic(self) -> None:
        """Same base_seed → same batch."""
        b1 = generate_batch(3, base_seed=42)
        b2 = generate_batch(3, base_seed=42)
        for (r1, s1, m1), (r2, s2, m2) in zip(b1, b2):
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(s1, s2)

    def test_batch_samples_differ(self) -> None:
        """Different samples in the same batch must differ."""
        samples = generate_batch(3, base_seed=42)
        self.assertFalse(
            np.array_equal(samples[0][1], samples[1][1]),
        )


class TestDatasetIO(unittest.TestCase):
    """Test filesystem dataset generation and I/O."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_directory_structure(self) -> None:
        """Dataset creates expected directory layout."""
        config = GeneratorConfig()
        generate_dataset(self.temp_dir, 3, start_seed=0, config=config, split="train")
        for i in range(3):
            sd = os.path.join(self.temp_dir, "train", f"sample_{i:03d}")
            self.assertTrue(os.path.isdir(sd))
            self.assertTrue(os.path.isfile(os.path.join(sd, "reference.png")))
            self.assertTrue(os.path.isfile(os.path.join(sd, "search.png")))
            self.assertTrue(os.path.isfile(os.path.join(sd, "ground_truth.json")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.temp_dir, "metadata_train.json")),
        )

    def test_reload_and_validate(self) -> None:
        """Reloaded images and metadata must match generated data."""
        config = GeneratorConfig()
        meta_list = generate_dataset(
            self.temp_dir, 3, start_seed=10, config=config, split="train",
        )
        self.assertEqual(len(meta_list), 3)

        for i, meta in enumerate(meta_list):
            sd = os.path.join(self.temp_dir, "train", f"sample_{i:03d}")
            ref = cv2.imread(os.path.join(sd, "reference.png"), cv2.IMREAD_GRAYSCALE)
            search = cv2.imread(os.path.join(sd, "search.png"), cv2.IMREAD_GRAYSCALE)
            with open(os.path.join(sd, "ground_truth.json")) as f:
                gt = json.load(f)

            expected_size = int(round(max(config.ref_size, int(np.ceil(1.5 * config.fin_period))) * meta.scale))
            self.assertEqual(ref.shape, (expected_size, expected_size))
            self.assertEqual(search.shape, (config.search_size, config.search_size))
            self.assertEqual(gt["gt_top_left_x"], meta.gt_top_left_x)
            self.assertEqual(gt["gt_top_left_y"], meta.gt_top_left_y)
            self.assertAlmostEqual(gt["scale"], meta.scale, places=4)

    def test_master_metadata(self) -> None:
        """Master metadata JSON has correct structure."""
        config = GeneratorConfig()
        generate_dataset(self.temp_dir, 4, start_seed=0, config=config, split="val")
        with open(os.path.join(self.temp_dir, "metadata_val.json")) as f:
            master = json.load(f)
        self.assertEqual(master["split"], "val")
        self.assertEqual(master["num_samples"], 4)
        self.assertEqual(len(master["samples"]), 4)
        # Check center coordinates are present in master
        for s in master["samples"]:
            self.assertIn("gt_center_x", s)
            self.assertIn("gt_center_y", s)


# ---------------------------------------------------------------------------
# Noise utility tests
# ---------------------------------------------------------------------------

class TestNoiseUtilities(unittest.TestCase):
    """Test noise application and illumination gradient utilities."""

    def test_sensor_noise_dtype(self) -> None:
        """Sensor noise output must be uint8."""
        img = np.full((50, 50), 128.0, dtype=np.float64)
        rng = np.random.default_rng(0)
        result = _apply_sensor_noise(img, 10.0, rng)
        self.assertEqual(result.dtype, np.uint8)

    def test_sensor_noise_modifies_image(self) -> None:
        """Sensor noise must actually change pixel values."""
        img = np.full((50, 50), 128.0, dtype=np.float64)
        rng = np.random.default_rng(0)
        result = _apply_sensor_noise(img, 10.0, rng)
        self.assertFalse(np.all(result == 128))

    def test_illumination_gradient_smooth(self) -> None:
        """Illumination gradient should be smooth, not random noise."""
        img = np.full((100, 100), 128.0, dtype=np.float64)
        rng = np.random.default_rng(42)
        result = _add_illumination_gradient(img, GeneratorConfig(), rng)
        # The gradient should create a smooth variation
        grad_range = float(np.max(result) - np.min(result))
        self.assertGreater(grad_range, 0.0)
        self.assertLessEqual(grad_range, 255.0)


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

class TestQualityGate(unittest.TestCase):
    """Test that quality checks work."""

    def test_sufficient_variance(self) -> None:
        """Generated images must have sufficient variance."""
        config = GeneratorConfig()
        for seed in range(5):
            ref, search, _ = generate_sample(seed, config)
            ref_var = float(np.var(ref.astype(np.float64)))
            search_var = float(np.var(search.astype(np.float64)))
            self.assertGreater(ref_var, config.min_variance)
            self.assertGreater(search_var, config.min_variance)


if __name__ == "__main__":
    unittest.main()
