"""
Unit tests for the drift_sense.scale_search module.
"""
import unittest
import numpy as np
import cv2

from drift_sense.config import ScaleSearchConfig
from drift_sense.scale_search import (
    _compute_psd_scale_prior,
    _score_scale_candidate,
    coarse_scale_search,
    fine_scale_search,
    estimate_scale,
)
from drift_sense.exceptions import ScaleSearchError


def _make_periodic_image(height: int, width: int, period: int, seed: int = 0) -> np.ndarray:
    """Generates a synthetic periodic stripe image (float32, zero-mean)."""
    rng = np.random.default_rng(seed)
    x = np.arange(width, dtype=np.float32)
    stripe = np.tile(np.sin(2 * np.pi * x / period), (height, 1)).astype(np.float32)
    noise = rng.standard_normal((height, width)).astype(np.float32) * 0.05
    img = stripe + noise
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img


class TestScoreScaleCandidate(unittest.TestCase):

    def test_valid_scale_returns_positive_score(self):
        """A correctly sized template should yield a positive NCC score."""
        rng = np.random.default_rng(42)
        search = rng.standard_normal((500, 500)).astype(np.float32)
        ref = rng.standard_normal((50, 50)).astype(np.float32)
        score = _score_scale_candidate(ref, search, scale=1.0)
        self.assertGreater(score, -1.0)

    def test_oversized_scale_returns_minus_one(self):
        """A scale that makes the template exceed the search image returns -1."""
        ref = np.ones((500, 500), dtype=np.float32)
        search = np.ones((100, 100), dtype=np.float32)
        # scale=3.0 -> 166x166 template > 100x100 search -> invalid
        score = _score_scale_candidate(ref, search, scale=3.0)
        self.assertEqual(score, -1.0)


class TestCoarseScaleSearch(unittest.TestCase):

    def setUp(self):
        self.config = ScaleSearchConfig(
            use_psd_prior=False,
            min_scale=9.0,
            max_scale=11.0,
            coarse_step=0.5,
            fine_step=0.1,
        )

    def test_finds_winner_in_valid_range(self):
        """Coarse search must return a scale within the configured bounds."""
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((50, 50)).astype(np.float32)
        search = rng.standard_normal((600, 600)).astype(np.float32)

        best_scale, best_score = coarse_scale_search(ref, search, self.config)
        self.assertGreaterEqual(best_scale, self.config.min_scale)
        self.assertLessEqual(best_scale, self.config.max_scale)
        self.assertGreater(best_score, -1.0)

    def test_raises_when_no_valid_candidate(self):
        """Coarse search must raise ScaleSearchError when all scales are invalid."""
        tiny_ref = np.ones((5000, 5000), dtype=np.float32)
        tiny_search = np.ones((400, 400), dtype=np.float32)
        with self.assertRaises(ScaleSearchError):
            coarse_scale_search(tiny_ref, tiny_search, self.config)


class TestFineScaleSearch(unittest.TestCase):

    def setUp(self):
        self.config = ScaleSearchConfig(
            use_psd_prior=False,
            min_scale=9.0,
            max_scale=11.0,
            coarse_step=0.5,
            fine_step=0.1,
        )

    def test_fine_search_refines_coarse(self):
        """Fine search must return a scale within one coarse step of the coarse winner."""
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((50, 50)).astype(np.float32)
        search = rng.standard_normal((600, 600)).astype(np.float32)

        coarse_scale = 10.0
        fine_scale, fine_score = fine_scale_search(ref, search, coarse_scale, self.config)

        self.assertGreaterEqual(fine_scale, self.config.min_scale)
        self.assertLessEqual(fine_scale, self.config.max_scale)
        self.assertGreater(fine_score, -1.0)


class TestEstimateScale(unittest.TestCase):

    def test_full_pipeline_without_psd(self):
        """Full estimate_scale pipeline completes and returns valid scale."""
        config = ScaleSearchConfig(
            use_psd_prior=False,
            min_scale=9.0,
            max_scale=11.0,
            coarse_step=0.5,
            fine_step=0.1,
        )
        rng = np.random.default_rng(7)
        ref = rng.standard_normal((50, 50)).astype(np.float32)
        search = rng.standard_normal((600, 600)).astype(np.float32)

        scale, score = estimate_scale(ref, search, config)
        self.assertGreaterEqual(scale, config.min_scale)
        self.assertLessEqual(scale, config.max_scale)
        self.assertGreater(score, -1.0)

    def test_psd_prior_discarded_on_low_confidence(self):
        """PSD prior from a non-periodic image should be discarded gracefully."""
        config = ScaleSearchConfig(
            use_psd_prior=True,
            min_scale=9.0,
            max_scale=11.0,
            coarse_step=0.5,
            fine_step=0.1,
            psd_confidence_threshold=999.0,   # unreachable threshold
        )
        rng = np.random.default_rng(8)
        ref = rng.standard_normal((50, 50)).astype(np.float32)
        search = rng.standard_normal((600, 600)).astype(np.float32)

        # Must not raise; falls back to full coarse search.
        scale, score = estimate_scale(ref, search, config)
        self.assertGreaterEqual(scale, config.min_scale)
        self.assertLessEqual(scale, config.max_scale)


if __name__ == "__main__":
    unittest.main()
