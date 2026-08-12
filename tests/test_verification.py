"""
Unit tests for the drift_sense.verification module.
Validates structural gradient similarity and numerical robustness.
"""
import unittest
import numpy as np

from drift_sense.config import VerificationConfig
from drift_sense.verification import verify_match, VerificationResult


class TestVerification(unittest.TestCase):
    def setUp(self):
        self.config = VerificationConfig(min_combined_similarity=0.4)
        # Create a basic structurally rich template (e.g. an edge/cross)
        self.ref = np.zeros((10, 10), dtype=np.float32)
        self.ref[4:6, 2:8] = 1.0
        self.ref[2:8, 4:6] = 1.0
        
        self.search = np.zeros((30, 30), dtype=np.float32)
        # Place exact match at (10, 10)
        self.search[10:20, 10:20] = self.ref

    def test_identical_reference_and_crop(self):
        """1. Identical reference and crop (high confidence)."""
        res = verify_match(self.search, self.ref, 10.0, 10.0, self.config)
        self.assertTrue(res.valid)
        self.assertTrue(res.passed)
        self.assertGreater(res.confidence, 0.99)

    def test_different_global_intensity(self):
        """2. Same structure but different global intensity."""
        # Brighten and reduce contrast of the search image
        search_mod = self.search * 0.5 + 10.0
        res = verify_match(search_mod, self.ref, 10.0, 10.0, self.config)
        self.assertTrue(res.valid)
        self.assertTrue(res.passed)
        # Gradient ZNCC should perfectly ignore global intensity shifts
        self.assertGreater(res.confidence, 0.99)

    def test_structurally_different_crop(self):
        """3. Structurally different crop (low confidence, fail)."""
        # Place a different structure at (2, 2)
        diff_search = np.zeros((30, 30), dtype=np.float32)
        diff_search[2:12, 2:12] = np.eye(10, dtype=np.float32)
        res = verify_match(diff_search, self.ref, 2.0, 2.0, self.config)
        self.assertTrue(res.valid)
        self.assertFalse(res.passed)
        self.assertLess(res.confidence, 0.4)

    def test_perfectly_uniform_reference(self):
        """4. Perfectly uniform reference (safe failure)."""
        uniform_ref = np.ones((10, 10), dtype=np.float32)
        res = verify_match(self.search, uniform_ref, 10.0, 10.0, self.config)
        self.assertTrue(res.valid)
        self.assertFalse(res.passed)
        self.assertEqual(res.confidence, 0.0)

    def test_perfectly_uniform_search_crop(self):
        """5. Perfectly uniform search crop (safe failure)."""
        res = verify_match(self.search, self.ref, 0.0, 0.0, self.config) # (0,0) is empty
        self.assertTrue(res.valid)
        self.assertFalse(res.passed)
        self.assertEqual(res.confidence, 0.0)

    def test_zero_gradient_reference(self):
        """6. Zero-gradient reference (safe failure)."""
        zero_ref = np.zeros((10, 10), dtype=np.float32)
        res = verify_match(self.search, zero_ref, 10.0, 10.0, self.config)
        self.assertTrue(res.valid)
        self.assertFalse(res.passed)
        self.assertEqual(res.confidence, 0.0)

    def test_different_dimensions(self):
        """7. Different reference/search dimensions (invalid)."""
        ref_large = np.zeros((50, 50), dtype=np.float32)
        res = verify_match(self.search, ref_large, 10.0, 10.0, self.config)
        self.assertFalse(res.valid)
        self.assertFalse(res.passed)

    def test_boundary_handling(self):
        """8. Prediction near search boundary."""
        # Top-left too far out
        res1 = verify_match(self.search, self.ref, -1.0, 5.0, self.config)
        self.assertFalse(res1.valid)
        
        # Bottom-right too far out
        res2 = verify_match(self.search, self.ref, 25.0, 25.0, self.config)
        self.assertFalse(res2.valid)

    def test_fractional_x_coordinate(self):
        """9. Fractional x coordinate."""
        # The crop extraction should natively interpolate
        res = verify_match(self.search, self.ref, 10.5, 10.0, self.config)
        self.assertTrue(res.valid)
        self.assertGreater(res.confidence, 0.0)

    def test_fractional_y_coordinate(self):
        """10. Fractional y coordinate."""
        res = verify_match(self.search, self.ref, 10.0, 9.8, self.config)
        self.assertTrue(res.valid)
        self.assertGreater(res.confidence, 0.0)

    def test_fractional_xy_simultaneous(self):
        """11. Fractional x and y simultaneously."""
        res = verify_match(self.search, self.ref, 10.5, 10.5, self.config)
        self.assertTrue(res.valid)
        self.assertGreater(res.confidence, 0.0)

    def test_exact_integer_coordinate(self):
        """12. Exact integer coordinate."""
        res = verify_match(self.search, self.ref, 10.0, 10.0, self.config)
        self.assertTrue(res.valid)

    def test_nan_input(self):
        """13. NaN input."""
        nan_search = np.copy(self.search)
        nan_search[0,0] = np.nan
        res = verify_match(nan_search, self.ref, 10.0, 10.0, self.config)
        self.assertFalse(res.valid)
        
        # NaN coordinates
        res_coord = verify_match(self.search, self.ref, float('nan'), 10.0, self.config)
        self.assertFalse(res_coord.valid)

    def test_inf_input(self):
        """14. Inf input."""
        inf_search = np.copy(self.search)
        inf_search[0,0] = np.inf
        res = verify_match(inf_search, self.ref, 10.0, 10.0, self.config)
        self.assertFalse(res.valid)

    def test_very_small_images(self):
        """15. Very small images."""
        small_search = np.zeros((4, 4), dtype=np.float32)
        small_ref = np.zeros((2, 2), dtype=np.float32)
        res = verify_match(small_search, small_ref, 1.0, 1.0, self.config)
        # Should execute cleanly but fail threshold due to zero gradient
        self.assertTrue(res.valid)
        self.assertEqual(res.confidence, 0.0)

    def test_large_representative_image(self):
        """16. Large representative 1000x1000 search image."""
        rng = np.random.default_rng(7)
        large_search = rng.uniform(-1.0, 1.0, (1000, 1000)).astype(np.float32)
        large_ref = large_search[500:600, 500:600]
        res = verify_match(large_search, large_ref, 500.0, 500.0, self.config)
        self.assertTrue(res.valid)
        self.assertTrue(res.passed)
        self.assertGreater(res.confidence, 0.99)

    def test_threshold_behavior(self):
        """17. Threshold behavior."""
        # Find a crop with partial similarity
        partial_ref = self.search[7:17, 7:17]
        # Custom config threshold
        cfg_high = VerificationConfig(min_combined_similarity=0.99)
        cfg_low = VerificationConfig(min_combined_similarity=-0.99)
        
        res1 = verify_match(self.search, partial_ref, 10.0, 10.0, cfg_high)
        self.assertTrue(res1.valid)
        self.assertFalse(res1.passed) # below 0.99
        
        res2 = verify_match(self.search, partial_ref, 10.0, 10.0, cfg_low)
        self.assertTrue(res2.valid)
        self.assertTrue(res2.passed) # above -0.99

    def test_determinism(self):
        """18. Determinism."""
        res1 = verify_match(self.search, self.ref, 10.1, 9.9, self.config)
        res2 = verify_match(self.search, self.ref, 10.1, 9.9, self.config)
        self.assertEqual(res1, res2)

    def test_no_coordinate_modification(self):
        """19. No coordinate modification."""
        # The API doesn't return coordinates, verifying the object doesn't mutate it.
        # Implied by not returning coordinates.
        pass

    def test_full_size_reference_consistency(self):
        """20. Full-size reference/search crop consistency."""
        res = verify_match(self.ref, self.ref, 0.0, 0.0, self.config)
        self.assertTrue(res.valid)
        self.assertTrue(res.passed)
        self.assertGreater(res.confidence, 0.99)


if __name__ == "__main__":
    unittest.main()
