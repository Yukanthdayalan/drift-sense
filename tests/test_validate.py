"""
Unit tests for the drift_sense.validate module.
"""
import unittest
import os
import tempfile
import numpy as np

from drift_sense.validate import (
    validate_image_path,
    validate_image_array,
    validate_reference_search_pairing
)
from drift_sense.exceptions import ImageLoadError, PreprocessingError

class TestValidate(unittest.TestCase):
    
    def test_validate_image_path_success(self):
        """Verify a valid file path passes validation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_name = tmp.name
        try:
            validate_image_path(tmp_name)
        finally:
            os.remove(tmp_name)
            
    def test_validate_image_path_failures(self):
        """Verify non-existent or invalid paths raise ImageLoadError."""
        with self.assertRaises(ImageLoadError):
            validate_image_path(123)  # type: ignore
            
        with self.assertRaises(ImageLoadError):
            validate_image_path("non_existent_fake_image_xyz.png")
            
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ImageLoadError):
                validate_image_path(tmpdir)
                
    def test_validate_image_array_success(self):
        """Verify correctly shaped numpy arrays pass."""
        valid_2d = np.zeros((100, 100), dtype=np.uint8)
        valid_3d = np.zeros((100, 100, 3), dtype=np.uint8)
        validate_image_array(valid_2d)
        validate_image_array(valid_3d)
        
    def test_validate_image_array_failures(self):
        """Verify invalid arrays raise PreprocessingError."""
        with self.assertRaises(PreprocessingError):
            validate_image_array([1, 2, 3])  # type: ignore
            
        with self.assertRaises(PreprocessingError):
            validate_image_array(np.array([]))
            
        with self.assertRaises(PreprocessingError):
            validate_image_array(np.zeros((100, 100, 3, 4)))
            
        with self.assertRaises(PreprocessingError):
            validate_image_array(np.zeros((5, 5)))
            
    def test_validate_reference_search_pairing(self):
        """Verify pairing logic correctly enforces sizing bounds."""
        ref = np.zeros((50, 50))
        search = np.zeros((500, 500))
        
        # Should pass
        validate_reference_search_pairing(ref, search)
        
        # Should fail: Reference larger than search
        with self.assertRaises(PreprocessingError):
            validate_reference_search_pairing(search, ref)
            
        # Should fail: Reference equal to search
        with self.assertRaises(PreprocessingError):
            validate_reference_search_pairing(ref, ref)

if __name__ == '__main__':
    unittest.main()
