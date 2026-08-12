"""
Unit tests for the drift_sense.utils module.
"""
import unittest

from drift_sense.types import Coordinate
from drift_sense.utils import calculate_distance, clamp_value, is_within_bounds

class TestUtils(unittest.TestCase):
    
    def test_calculate_distance(self):
        """Verify Euclidean distance computation is correct."""
        p1 = Coordinate(0.0, 0.0)
        p2 = Coordinate(3.0, 4.0)
        self.assertAlmostEqual(calculate_distance(p1, p2), 5.0)
        
    def test_clamp_value(self):
        """Verify clamping handles boundaries and out-of-bounds correctly."""
        self.assertEqual(clamp_value(5.0, 0.0, 10.0), 5.0)
        self.assertEqual(clamp_value(-5.0, 0.0, 10.0), 0.0)
        self.assertEqual(clamp_value(15.0, 0.0, 10.0), 10.0)
        
    def test_clamp_value_invalid_bounds(self):
        """Verify clamp_value raises ValueError when bounds are reversed."""
        with self.assertRaises(ValueError):
            clamp_value(5.0, 10.0, 0.0)
            
    def test_is_within_bounds(self):
        """Verify bounds checking logic with and without margins."""
        coord = Coordinate(5.0, 5.0)
        self.assertTrue(is_within_bounds(coord, 10, 10))
        self.assertFalse(is_within_bounds(coord, 10, 10, margin=6.0))
        
        out_coord = Coordinate(11.0, 5.0)
        self.assertFalse(is_within_bounds(out_coord, 10, 10))

    def test_is_within_bounds_negative_margin(self):
        """Verify bounds check raises error on negative margin."""
        coord = Coordinate(5.0, 5.0)
        with self.assertRaises(ValueError):
            is_within_bounds(coord, 10, 10, margin=-1.0)

if __name__ == '__main__':
    unittest.main()
