import os
import tempfile
import shutil
import unittest
import subprocess
import json
import cv2

from drift_sense.dataset import generate_sample, GeneratorConfig

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate one sample for testing
        self.config = GeneratorConfig(noise_sigma_search=3.0, noise_sigma_ref=5.0)
        self.ref, self.search, self.meta = generate_sample(42, self.config)
        
        self.ref_path = os.path.join(self.temp_dir, "ref.png")
        self.search_path = os.path.join(self.temp_dir, "search.png")
        self.out_path = os.path.join(self.temp_dir, "output.json")
        
        cv2.imwrite(self.ref_path, self.ref)
        cv2.imwrite(self.search_path, self.search)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_inference(self):
        # Explicitly run the root inference.py
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        
        cmd = ["python", "inference.py", self.ref_path, self.search_path, "--output", self.out_path]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        
        # Output should be valid JSON in the file
        self.assertTrue(os.path.exists(self.out_path))
        with open(self.out_path, "r") as f:
            data = json.load(f)
            
        self.assertIn("prediction_x", data)
        self.assertIn("prediction_y", data)
        
        x = data["prediction_x"]
        y = data["prediction_y"]
        
        # Check it is finite and within bounds
        self.assertTrue(0 <= x <= 1000)
        self.assertTrue(0 <= y <= 1000)

if __name__ == "__main__":
    unittest.main()
