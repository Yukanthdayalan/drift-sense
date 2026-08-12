import unittest
import os
import tempfile
import json
import numpy as np
import cv2

from drift_sense.inference import main, run_inference, run_batch_inference
from drift_sense.types import InferenceResult

class TestInference(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple valid image pair for testing
        s_img = np.zeros((100, 100), dtype=np.uint8)
        r_img = np.zeros((10, 10), dtype=np.uint8)
        
        s_img[20:30, 20:30] = 255
        r_img[:, :] = 255
        
        self.s_path = os.path.join(self.temp_dir.name, "search.png")
        self.r_path = os.path.join(self.temp_dir.name, "ref.png")
        
        cv2.imwrite(self.s_path, s_img)
        cv2.imwrite(self.r_path, r_img)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_run_inference(self):
        res = run_inference(self.r_path, self.s_path)
        self.assertIsInstance(res, InferenceResult)
        
    def test_run_batch_inference(self):
        pairs = [(self.r_path, self.s_path), (self.r_path, self.s_path)]
        results = run_batch_inference(pairs)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], InferenceResult)
        self.assertIsInstance(results[1], InferenceResult)
        
    def test_main_cli_success(self):
        """Test main CLI entry point."""
        out_json = os.path.join(self.temp_dir.name, "out.json")
        ret = main([self.r_path, self.s_path, "--output", out_json])
        self.assertEqual(ret, 0)
        
        self.assertTrue(os.path.exists(out_json))
        with open(out_json, "r") as f:
            data = json.load(f)
            
        self.assertIn("prediction_x", data)
        self.assertIn("prediction_y", data)
        self.assertIn("scale_used", data)
        
    def test_main_cli_invalid_image(self):
        ret = main(["non_existent.png", self.s_path])
        self.assertEqual(ret, 1)

if __name__ == "__main__":
    unittest.main()
