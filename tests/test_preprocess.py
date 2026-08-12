"""
Unit tests for the drift_sense.preprocess module.
"""
import unittest
import os
import tempfile
import cv2
import numpy as np

from drift_sense.preprocess import (
    load_image,
    apply_clahe,
    apply_gaussian_blur,
    apply_median_blur,
    z_score_normalize,
    preprocess_image
)
from drift_sense.config import PreprocessingConfig
from drift_sense.exceptions import ImageLoadError

class TestPreprocess(unittest.TestCase):
    
    def setUp(self):
        """Generates a synthetic noisy test image on disk."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img_path = os.path.join(self.temp_dir.name, "test_img.png")
        
        # Create a synthetic 50x50 noisy gradient image
        gradient = np.tile(np.linspace(0, 255, 50, dtype=np.uint8), (50, 1))
        np.random.seed(42)  # Deterministic noise
        noise = np.random.randint(0, 50, (50, 50), dtype=np.uint8)
        self.synthetic_img = cv2.add(gradient, noise)
        cv2.imwrite(self.img_path, self.synthetic_img)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_load_image(self):
        """Verify image loads as grayscale uint8."""
        img = load_image(self.img_path)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape, (50, 50))
        
    def test_load_image_invalid_file(self):
        """Verify loading non-image file raises ImageLoadError."""
        bad_path = os.path.join(self.temp_dir.name, "bad.txt")
        with open(bad_path, 'w') as f:
            f.write("Not an image.")
            
        with self.assertRaises(ImageLoadError):
            load_image(bad_path)
            
    def test_apply_clahe(self):
        """Verify CLAHE applies without crashing and preserves types."""
        img = load_image(self.img_path)
        clahe_img = apply_clahe(img)
        self.assertEqual(clahe_img.dtype, np.uint8)
        self.assertEqual(clahe_img.shape, (50, 50))
        
    def test_apply_gaussian_blur(self):
        """Verify Gaussian blur suppresses high-frequency variance."""
        img = load_image(self.img_path)
        blurred = apply_gaussian_blur(img, kernel_size=(5, 5), sigma=1.0)
        self.assertTrue(np.var(blurred) < np.var(img))
        
    def test_apply_median_blur(self):
        """Verify Median blur executes correctly."""
        img = load_image(self.img_path)
        blurred = apply_median_blur(img, kernel_size=3)
        self.assertEqual(blurred.shape, (50, 50))

    def test_z_score_normalize(self):
        """Verify Z-score mathematically normalizes to N(0, 1)."""
        img = load_image(self.img_path)
        norm_img = z_score_normalize(img)
        
        self.assertEqual(norm_img.dtype, np.float32)
        self.assertAlmostEqual(np.mean(norm_img), 0.0, places=5)
        self.assertAlmostEqual(np.std(norm_img), 1.0, places=5)
        
    def test_z_score_normalize_zero_variance(self):
        """Verify flat-line images are handled without ZeroDivisionError."""
        flat_img = np.ones((50, 50), dtype=np.uint8) * 128
        norm_img = z_score_normalize(flat_img)
        self.assertTrue(np.all(norm_img == 0.0))
        
    def test_preprocess_image_integration(self):
        """Verify the full pipeline orchestration works perfectly."""
        config = PreprocessingConfig(
            apply_clahe=True,
            apply_gaussian_blur=True,
            gaussian_kernel_size=(3, 3)
        )
        final_img = preprocess_image(self.img_path, config)
        
        self.assertEqual(final_img.dtype, np.float32)
        self.assertAlmostEqual(np.mean(final_img), 0.0, places=5)
        self.assertAlmostEqual(np.std(final_img), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
