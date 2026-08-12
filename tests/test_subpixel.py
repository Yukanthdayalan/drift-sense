"""
Unit tests for the drift_sense.subpixel module.
Validates the mathematical correctness and numerical stability of parabolic interpolation.
"""
import unittest
import numpy as np

from drift_sense.config import SubpixelConfig
from drift_sense.subpixel import refine_subpixel, SubpixelResult


class TestSubpixelRefinement(unittest.TestCase):
    def setUp(self):
        self.config = SubpixelConfig(max_offset=0.5)

    def test_perfect_symmetric_parabolic_peak_x(self):
        """1. Perfect symmetric parabolic peak along X."""
        # y = -a(x - h)^2 + k => h = 10.25, a = 1.0, k = 1.0
        # z(-1) = z(9) = -(9 - 10.25)^2 + 1 = -1.5625 + 1 = -0.5625
        # z(0) = z(10) = -(10 - 10.25)^2 + 1 = -0.0625 + 1 = 0.9375
        # z(+1) = z(11) = -(11 - 10.25)^2 + 1 = -0.5625 + 1 = 0.4375
        m = np.zeros((20, 20), dtype=np.float32)
        m[10, 9] = -0.5625
        m[10, 10] = 0.9375
        m[10, 11] = 0.4375
        
        # In Y, symmetric
        m[9, 10] = 0.9375
        m[11, 10] = 0.9375
        
        res = refine_subpixel(m, 10, 10, self.config)
        self.assertAlmostEqual(res.dx, 0.25, places=5)
        self.assertAlmostEqual(res.dy, 0.0, places=5)
        self.assertAlmostEqual(res.x, 10.25, places=5)
        self.assertAlmostEqual(res.y, 10.0, places=5)
        self.assertTrue(res.refined)

    def test_peak_exactly_integer(self):
        """2. Peak exactly at integer coordinate."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 0.5
        m[2, 2] = 1.0
        m[2, 3] = 0.5
        m[1, 2] = 0.5
        m[3, 2] = 0.5
        
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertAlmostEqual(res.dx, 0.0, places=5)
        self.assertAlmostEqual(res.dy, 0.0, places=5)
        self.assertAlmostEqual(res.x, 2.0, places=5)
        self.assertAlmostEqual(res.y, 2.0, places=5)
        self.assertFalse(res.refined)

    def test_positive_subpixel_offset(self):
        """3. Positive sub-pixel offset."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 0.3
        m[2, 2] = 1.0
        m[2, 3] = 0.8
        # Since z_+1 > z_-1, the peak shifts towards +1 (positive offset)
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertGreater(res.dx, 0.0)
        self.assertTrue(res.refined)

    def test_negative_subpixel_offset(self):
        """4. Negative sub-pixel offset."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 0.8
        m[2, 2] = 1.0
        m[2, 3] = 0.3
        # Since z_-1 > z_+1, the peak shifts towards -1 (negative offset)
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertLess(res.dx, 0.0)
        self.assertTrue(res.refined)

    def test_x_refinement_y_unchanged(self):
        """5. X refinement while Y remains unchanged."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 0.5
        m[2, 2] = 1.0
        m[2, 3] = 0.7  # X shifts
        m[1, 2] = 0.5
        m[3, 2] = 0.5  # Y balanced
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertNotEqual(res.dx, 0.0)
        self.assertEqual(res.dy, 0.0)

    def test_y_refinement_x_unchanged(self):
        """6. Y refinement while X remains unchanged."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 0.5
        m[2, 2] = 1.0
        m[2, 3] = 0.5  # X balanced
        m[1, 2] = 0.4
        m[3, 2] = 0.8  # Y shifts
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertEqual(res.dx, 0.0)
        self.assertNotEqual(res.dy, 0.0)

    def test_2d_subpixel_peak(self):
        """7. Two-dimensional sub-pixel peak."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 0.2
        m[2, 2] = 1.0
        m[2, 3] = 0.9
        m[1, 2] = 0.9
        m[3, 2] = 0.2
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertGreater(res.dx, 0.0)
        self.assertLess(res.dy, 0.0)
        self.assertTrue(res.refined)

    def test_boundary_peak(self):
        """8. Boundary peak (fallback to integer)."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[0, 2] = 1.0
        res = refine_subpixel(m, 2, 0, self.config)
        self.assertEqual(res.dx, 0.0)
        self.assertEqual(res.dy, 0.0)
        self.assertFalse(res.refined)

    def test_flat_response(self):
        """9. Flat response around the peak."""
        m = np.ones((5, 5), dtype=np.float32)
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertEqual(res.dx, 0.0)
        self.assertEqual(res.dy, 0.0)
        self.assertFalse(res.refined)

    def test_near_zero_denominator(self):
        """10. Near-zero second derivative."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 1.0
        m[2, 2] = 1.0
        m[2, 3] = 1.0
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertEqual(res.dx, 0.0)

    def test_nan_neighboring_value(self):
        """11. NaN neighboring value."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = float('nan')
        m[2, 2] = 1.0
        m[2, 3] = 0.5
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertEqual(res.dx, 0.0)

    def test_inf_neighboring_value(self):
        """12. Inf neighboring value."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = float('inf')
        m[2, 2] = 1.0
        m[2, 3] = 0.5
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertEqual(res.dx, 0.0)

    def test_offset_larger_than_max(self):
        """13. Offset larger than 0.5 (clamping check)."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = -1.0
        m[2, 2] = 1.0
        m[2, 3] = 2.0 
        # delta calculation = (-1.0 - 2.0) / 2(-1.0 - 2.0 + 2.0) = -3.0 / -2.0 = 1.5
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertAlmostEqual(res.dx, 0.5, places=5)
        self.assertTrue(res.refined)

    def test_offset_smaller_than_min(self):
        """14. Offset smaller than -0.5 (clamping check)."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 2.0
        m[2, 2] = 1.0
        m[2, 3] = -1.0
        # delta calculation = (2.0 - -1.0) / 2(2.0 - 2.0 - 1.0) = 3.0 / -2.0 = -1.5
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertAlmostEqual(res.dx, -0.5, places=5)
        self.assertTrue(res.refined)

    def test_determinism(self):
        """15. Determinism."""
        m = np.random.RandomState(42).rand(10, 10).astype(np.float32)
        res1 = refine_subpixel(m, 5, 5, self.config)
        res2 = refine_subpixel(m, 5, 5, self.config)
        self.assertEqual(res1.dx, res2.dx)
        self.assertEqual(res1.dy, res2.dy)

    def test_invalid_response_map_shape(self):
        """16. Invalid response-map shape."""
        with self.assertRaises(ValueError):
            refine_subpixel(np.zeros((5, 5, 3)), 2, 2, self.config)

    def test_invalid_peak_coordinates(self):
        """17. Invalid peak coordinates."""
        m = np.zeros((5, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            refine_subpixel(m, -1, 2, self.config)
        with self.assertRaises(ValueError):
            refine_subpixel(m, 5, 2, self.config)

    def test_float32_response_map(self):
        """18. Float32 response map."""
        m = np.zeros((5, 5), dtype=np.float32)
        m[2, 1] = 0.5; m[2, 2] = 1.0; m[2, 3] = 0.8
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertTrue(res.refined)

    def test_float64_response_map(self):
        """19. Float64 response map."""
        m = np.zeros((5, 5), dtype=np.float64)
        m[2, 1] = 0.5; m[2, 2] = 1.0; m[2, 3] = 0.8
        res = refine_subpixel(m, 2, 2, self.config)
        self.assertTrue(res.refined)

    def test_large_response_map(self):
        """20. Large response map representative of a 1000x1000 search image."""
        m = np.random.RandomState(7).rand(1000, 1000).astype(np.float32)
        m[500, 499] = 0.5
        m[500, 500] = 1.0
        m[500, 501] = 0.8
        res = refine_subpixel(m, 500, 500, self.config)
        self.assertTrue(res.refined)


if __name__ == "__main__":
    unittest.main()
