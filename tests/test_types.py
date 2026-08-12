"""
Unit tests for the drift_sense.types module.
"""
import unittest
import numpy as np

from drift_sense.types import Coordinate, MatchCandidate, InferenceResult, ImageArray

class TestTypes(unittest.TestCase):
    
    def test_coordinate_creation(self):
        """Verify Coordinate holds float values correctly."""
        coord = Coordinate(x=10.5, y=20.1)
        self.assertEqual(coord.x, 10.5)
        self.assertEqual(coord.y, 20.1)

    def test_match_candidate_defaults(self):
        """Verify MatchCandidate defaults are applied correctly."""
        cand = MatchCandidate(x=5.0, y=5.0, scale=10.0, ncc_score=0.95)
        self.assertEqual(cand.peak_ratio, 0.0)
        self.assertEqual(cand.distance_to_center, 0.0)
        self.assertEqual(cand.final_score, 0.0)
        self.assertEqual(cand.ncc_score, 0.95)

    def test_inference_result_instantiation(self):
        """Verify InferenceResult encapsulates final payload."""
        res = InferenceResult(
            prediction=Coordinate(x=100.0, y=100.0),
            scale_used=9.85,
            confidence=0.92,
            is_fallback_triggered=True,
            execution_time_ms=45.2
        )
        self.assertTrue(isinstance(res.prediction, Coordinate))
        self.assertEqual(res.message, "Success")
        self.assertEqual(res.candidates_found, 0)
        
    def test_image_array_alias(self):
        """Verify ImageArray alias acts as an ndarray hint."""
        img: ImageArray = np.zeros((10, 10), dtype=np.float32)
        self.assertIsInstance(img, np.ndarray)

if __name__ == '__main__':
    unittest.main()
