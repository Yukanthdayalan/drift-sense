"""
Unit tests for the drift_sense.exceptions module.
"""
import unittest

from drift_sense.exceptions import (
    DriftSenseError,
    ImageLoadError,
    PreprocessingError,
    ScaleSearchError,
    MatchingError,
    VerificationError
)

class TestExceptions(unittest.TestCase):
    
    def test_base_inheritance(self):
        """Verify all custom errors inherit from the base DriftSenseError."""
        self.assertTrue(issubclass(ImageLoadError, DriftSenseError))
        self.assertTrue(issubclass(PreprocessingError, DriftSenseError))
        self.assertTrue(issubclass(ScaleSearchError, DriftSenseError))
        self.assertTrue(issubclass(MatchingError, DriftSenseError))
        self.assertTrue(issubclass(VerificationError, DriftSenseError))
        
    def test_exception_catching(self):
        """Verify exceptions can be caught via the base class."""
        with self.assertRaises(DriftSenseError):
            raise ScaleSearchError("Scale out of bounds.")
            
        with self.assertRaises(MatchingError):
            raise MatchingError("Zero variance in ZNCC.")

if __name__ == '__main__':
    unittest.main()
