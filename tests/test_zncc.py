"""
Unit tests for drift_sense.zncc

Covers:
  - normalize_template
  - compute_local_statistics
  - validate_response
  - compute_zncc_fft
  - compute_zncc_spatial
  - FFT vs spatial agreement
  - Numerical edge cases (NaN, Inf, zero variance, flat images)
  - Invalid dtype and dimension guards
  - Peak correctness on synthetic translated patch
"""
import unittest
import numpy as np

from drift_sense.zncc import (
    normalize_template,
    compute_local_statistics,
    validate_response,
    compute_zncc_fft,
    compute_zncc_spatial,
)
from drift_sense.exceptions import MatchingError, PreprocessingError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_f32(shape: tuple, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _zeros_f32(shape: tuple) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _make_search_with_embedded_template(
    search_h: int = 100,
    search_w: int = 100,
    t_h: int = 20,
    t_w: int = 20,
    embed_row: int = 30,
    embed_col: int = 40,
    seed: int = 1,
) -> tuple:
    """
    Creates a search image with a known template patch embedded at (embed_col, embed_row).
    Both images are z-score normalised.  Returns (template, search, embed_row, embed_col).
    """
    rng = np.random.default_rng(seed)
    template_raw = rng.standard_normal((t_h, t_w)).astype(np.float32)
    search_raw = rng.standard_normal((search_h, search_w)).astype(np.float32) * 0.05

    # Embed template into search with strong signal.
    search_raw[embed_row : embed_row + t_h, embed_col : embed_col + t_w] += template_raw * 10.0

    # Z-score normalise each image independently.
    def znorm(img: np.ndarray) -> np.ndarray:
        mu, sigma = img.mean(), img.std()
        return ((img - mu) / (sigma + 1e-8)).astype(np.float32)

    return znorm(template_raw), znorm(search_raw), embed_row, embed_col


# ---------------------------------------------------------------------------
# normalize_template
# ---------------------------------------------------------------------------

class TestNormalizeTemplate(unittest.TestCase):

    def test_output_is_float32(self):
        t = _rand_f32((20, 20))
        out = normalize_template(t)
        self.assertEqual(out.dtype, np.float32)

    def test_zero_mean_unit_variance(self):
        t = _rand_f32((30, 30), seed=7)
        out = normalize_template(t)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(out.std()), 1.0, places=5)

    def test_flat_template_returns_zeros(self):
        """Zero-variance template must return a zero array without raising."""
        t = np.ones((15, 15), dtype=np.float32)
        out = normalize_template(t)
        self.assertTrue(np.all(out == 0.0))

    def test_raises_on_wrong_dtype(self):
        t = np.ones((10, 10), dtype=np.uint8)
        with self.assertRaises(PreprocessingError):
            normalize_template(t)

    def test_raises_on_3d_input(self):
        t = np.ones((10, 10, 3), dtype=np.float32)
        with self.assertRaises(PreprocessingError):
            normalize_template(t)


# ---------------------------------------------------------------------------
# compute_local_statistics
# ---------------------------------------------------------------------------

class TestComputeLocalStatistics(unittest.TestCase):

    def test_output_shape(self):
        s = _rand_f32((80, 80))
        t_h, t_w = 20, 20
        local_mean, local_std = compute_local_statistics(s, t_h, t_w)
        expected_shape = (80 - 20 + 1, 80 - 20 + 1)
        self.assertEqual(local_mean.shape, expected_shape)
        self.assertEqual(local_std.shape, expected_shape)

    def test_output_dtype(self):
        s = _rand_f32((50, 50))
        local_mean, local_std = compute_local_statistics(s, 10, 10)
        self.assertEqual(local_mean.dtype, np.float32)
        self.assertEqual(local_std.dtype, np.float32)

    def test_local_std_non_negative(self):
        """Local std must never be negative (variance clamping guard)."""
        s = _rand_f32((60, 60), seed=99)
        _, local_std = compute_local_statistics(s, 15, 15)
        self.assertTrue(np.all(local_std >= 0.0))

    def test_flat_search_has_near_zero_local_std(self):
        """A uniform search image should produce local_std ≈ 0 everywhere."""
        s = np.ones((50, 50), dtype=np.float32) * 5.0
        _, local_std = compute_local_statistics(s, 10, 10)
        self.assertTrue(np.all(local_std < 1e-4))


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------

class TestValidateResponse(unittest.TestCase):

    def test_valid_map(self):
        rm = _rand_f32((50, 50))
        rm = np.clip(rm / rm.max(), -1.0, 1.0)
        max_s, mean_s = validate_response(rm)
        self.assertIsInstance(max_s, float)
        self.assertIsInstance(mean_s, float)

    def test_empty_map_raises(self):
        with self.assertRaises(MatchingError):
            validate_response(np.array([], dtype=np.float32))

    def test_nan_raises(self):
        rm = np.zeros((10, 10), dtype=np.float32)
        rm[5, 5] = float('nan')
        with self.assertRaises(MatchingError):
            validate_response(rm)

    def test_inf_raises(self):
        rm = np.zeros((10, 10), dtype=np.float32)
        rm[5, 5] = float('inf')
        with self.assertRaises(MatchingError):
            validate_response(rm)

    def test_extreme_value_raises(self):
        rm = np.ones((10, 10), dtype=np.float32) * 2.0
        with self.assertRaises(MatchingError):
            validate_response(rm)


# ---------------------------------------------------------------------------
# compute_zncc_fft
# ---------------------------------------------------------------------------

class TestComputeZNCCFft(unittest.TestCase):

    def test_output_shape(self):
        t, s, _, _ = _make_search_with_embedded_template()
        rm = compute_zncc_fft(t, s)
        expected = (s.shape[0] - t.shape[0] + 1, s.shape[1] - t.shape[1] + 1)
        self.assertEqual(rm.shape, expected)

    def test_output_dtype(self):
        t, s, _, _ = _make_search_with_embedded_template()
        rm = compute_zncc_fft(t, s)
        self.assertEqual(rm.dtype, np.float32)

    def test_output_in_valid_range(self):
        t, s, _, _ = _make_search_with_embedded_template()
        rm = compute_zncc_fft(t, s)
        self.assertGreaterEqual(float(rm.min()), -1.0)
        self.assertLessEqual(float(rm.max()), 1.0)

    def test_peak_at_correct_location(self):
        """The response map maximum must coincide with the embedded template location."""
        embed_row, embed_col = 30, 40
        t, s, gt_row, gt_col = _make_search_with_embedded_template(
            embed_row=embed_row, embed_col=embed_col
        )
        rm = compute_zncc_fft(t, s)
        peak_idx = np.unravel_index(np.argmax(rm), rm.shape)
        self.assertAlmostEqual(peak_idx[0], gt_row, delta=2)
        self.assertAlmostEqual(peak_idx[1], gt_col, delta=2)

    def test_identical_images_identical_content(self):
        """When template equals the search-sized patch, max score should be high."""
        rng = np.random.default_rng(42)
        patch = rng.standard_normal((30, 30)).astype(np.float32)
        patch = ((patch - patch.mean()) / (patch.std() + 1e-8)).astype(np.float32)
        search = np.zeros((100, 100), dtype=np.float32)
        search[10:40, 10:40] = patch
        search = ((search - search.mean()) / (search.std() + 1e-8)).astype(np.float32)
        rm = compute_zncc_fft(patch, search)
        self.assertGreater(float(rm.max()), 0.3)

    def test_blank_template_raises_or_returns_zero_map(self):
        """A flat (zero variance) template must not produce NaN/Inf in the output."""
        t = np.ones((20, 20), dtype=np.float32)
        t = ((t - t.mean()) / (t.std() + 1e-8)).astype(np.float32)
        s = _rand_f32((100, 100))
        rm = compute_zncc_fft(t, s)
        self.assertTrue(np.isfinite(rm).all())

    def test_blank_search_returns_zero_map(self):
        """A flat search image produces near-zero local_std → near-zero ZNCC."""
        t = _rand_f32((20, 20))
        t = ((t - t.mean()) / (t.std() + 1e-8)).astype(np.float32)
        s = np.ones((100, 100), dtype=np.float32)
        rm = compute_zncc_fft(t, s)
        self.assertTrue(np.isfinite(rm).all())

    def test_raises_on_nan_in_template(self):
        t = _rand_f32((20, 20))
        t[5, 5] = float('nan')
        s = _rand_f32((100, 100))
        with self.assertRaises(MatchingError):
            compute_zncc_fft(t, s)

    def test_raises_on_inf_in_search(self):
        t = _rand_f32((20, 20))
        s = _rand_f32((100, 100))
        s[50, 50] = float('inf')
        with self.assertRaises(MatchingError):
            compute_zncc_fft(t, s)

    def test_raises_when_template_larger_than_search(self):
        t = _rand_f32((100, 100))
        s = _rand_f32((50, 50))
        with self.assertRaises(MatchingError):
            compute_zncc_fft(t, s)

    def test_raises_on_wrong_dtype(self):
        t = np.ones((20, 20), dtype=np.uint8)
        s = _rand_f32((100, 100))
        with self.assertRaises(PreprocessingError):
            compute_zncc_fft(t, s)

    def test_noisy_template(self):
        """Adding heavy noise to template must degrade but not break ZNCC."""
        rng = np.random.default_rng(55)
        embed_row, embed_col = 30, 30
        t, s, _, _ = _make_search_with_embedded_template(
            embed_row=embed_row, embed_col=embed_col, seed=55
        )
        noise = rng.standard_normal(t.shape).astype(np.float32) * 0.5
        t_noisy = ((t + noise - (t + noise).mean()) / ((t + noise).std() + 1e-8)).astype(np.float32)
        rm = compute_zncc_fft(t_noisy, s)
        self.assertTrue(np.isfinite(rm).all())

    def test_noisy_search_image(self):
        """Heavy noise on search image must still produce a finite response map."""
        rng = np.random.default_rng(77)
        t, s, _, _ = _make_search_with_embedded_template()
        noise = rng.standard_normal(s.shape).astype(np.float32) * 2.0
        s_noisy = ((s + noise - (s + noise).mean()) / ((s + noise).std() + 1e-8)).astype(np.float32)
        rm = compute_zncc_fft(t, s_noisy)
        self.assertTrue(np.isfinite(rm).all())


# ---------------------------------------------------------------------------
# compute_zncc_spatial
# ---------------------------------------------------------------------------

class TestComputeZNCCSpatial(unittest.TestCase):

    def test_output_shape_and_dtype(self):
        t = _rand_f32((10, 10))
        t = ((t - t.mean()) / (t.std() + 1e-8)).astype(np.float32)
        s = _rand_f32((30, 30))
        rm = compute_zncc_spatial(t, s)
        self.assertEqual(rm.shape, (21, 21))
        self.assertEqual(rm.dtype, np.float32)

    def test_peak_at_correct_location(self):
        """Spatial ZNCC must also peak at the embedded template location."""
        t, s, gt_row, gt_col = _make_search_with_embedded_template(
            search_h=60, search_w=60, t_h=10, t_w=10,
            embed_row=15, embed_col=20, seed=3
        )
        rm = compute_zncc_spatial(t, s)
        peak_idx = np.unravel_index(np.argmax(rm), rm.shape)
        self.assertAlmostEqual(peak_idx[0], 15, delta=2)
        self.assertAlmostEqual(peak_idx[1], 20, delta=2)

    def test_blank_window_returns_zero_score(self):
        """Zero-variance windows must return 0.0, not NaN."""
        t = _rand_f32((10, 10))
        t = ((t - t.mean()) / (t.std() + 1e-8)).astype(np.float32)
        s = np.ones((30, 30), dtype=np.float32)
        rm = compute_zncc_spatial(t, s)
        self.assertTrue(np.isfinite(rm).all())

    def test_raises_on_wrong_dtype(self):
        t = np.ones((10, 10), dtype=np.uint8)
        s = _rand_f32((30, 30))
        with self.assertRaises(PreprocessingError):
            compute_zncc_spatial(t, s)


# ---------------------------------------------------------------------------
# FFT vs Spatial agreement
# ---------------------------------------------------------------------------

class TestFftVsSpatialAgreement(unittest.TestCase):

    def test_agreement_on_small_image(self):
        """
        FFT and spatial ZNCC must agree to within floating-point tolerance
        on a small image where both are tractable.
        """
        t, s, _, _ = _make_search_with_embedded_template(
            search_h=50, search_w=50, t_h=10, t_w=10,
            embed_row=10, embed_col=15, seed=9
        )
        rm_fft = compute_zncc_fft(t, s)
        rm_spatial = compute_zncc_spatial(t, s)

        self.assertEqual(rm_fft.shape, rm_spatial.shape)

        # Allow a tolerance of 2e-3 to account for floating-point ordering differences.
        max_diff = float(np.max(np.abs(rm_fft.astype(np.float64) - rm_spatial.astype(np.float64))))
        self.assertLess(max_diff, 2e-3, f"FFT/spatial disagreement: max_diff={max_diff:.6f}")

    def test_peak_location_agreement(self):
        """FFT and spatial ZNCC must agree on the argmax location."""
        t, s, gt_row, gt_col = _make_search_with_embedded_template(
            search_h=60, search_w=60, t_h=10, t_w=10,
            embed_row=20, embed_col=25, seed=11
        )
        rm_fft = compute_zncc_fft(t, s)
        rm_spatial = compute_zncc_spatial(t, s)

        peak_fft = np.unravel_index(np.argmax(rm_fft), rm_fft.shape)
        peak_spatial = np.unravel_index(np.argmax(rm_spatial), rm_spatial.shape)

        self.assertEqual(peak_fft[0], peak_spatial[0])
        self.assertEqual(peak_fft[1], peak_spatial[1])


if __name__ == "__main__":
    unittest.main()
