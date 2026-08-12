"""
Unit tests for the drift_sense.config module.
Validates the instantiation and immutability of the configuration structures.
"""
import unittest
from dataclasses import FrozenInstanceError

from drift_sense.config import get_default_config, EngineConfig, ScaleSearchConfig

class TestConfig(unittest.TestCase):
    
    def test_default_config_creation(self):
        """Verify the default config can be instantiated with expected types."""
        config = get_default_config()
        self.assertIsInstance(config, EngineConfig)
        self.assertIsInstance(config.scale_search, ScaleSearchConfig)
        
        # Verify specific defaults according to specification
        self.assertEqual(config.scale_search.min_scale, 9.0)
        self.assertEqual(config.scale_search.max_scale, 11.0)
        self.assertEqual(config.tie_break.delta, 0.05)
        
    def test_config_is_immutable(self):
        """Verify the configuration objects are frozen and prevent runtime modification."""
        config = get_default_config()
        
        with self.assertRaises(FrozenInstanceError):
            config.scale_search.min_scale = 5.0
            
        with self.assertRaises(FrozenInstanceError):
            config.tie_break.delta = 0.1

if __name__ == '__main__':
    unittest.main()
