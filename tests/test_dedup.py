import pytest
import numpy as np
from drift_sense.scale_search import estimate_top_n_scales
from drift_sense.config import get_default_config

def test_scale_deduplication():
    config = get_default_config()
    
    # Create synthetic reference and search images
    ref = np.random.randn(20, 20).astype(np.float32)
    search = np.random.randn(100, 100).astype(np.float32)
    
    import drift_sense.scale_search as ss
    
    # Mock fine and ultra_fine to always return 10.0 for any input
    original_coarse = ss._score_scale_candidate
    original_fine = ss.fine_scale_search
    original_ultra = ss.ultra_fine_scale_search
    
    try:
        ss._score_scale_candidate = lambda r, s, sc: 0.8
        ss.fine_scale_search = lambda r, s, cs, c: (10.0, 0.9)
        ss.ultra_fine_scale_search = lambda r, s, fs, c: (10.0, 0.95)
        
        # In the buggy version, this would return [10.0, 10.0]
        # In the fixed version, this should return [10.0] and stop, or expand search
        scales = estimate_top_n_scales(ref, search, config.scale_search, n_scales=2)
        
        # Verify deduplication worked
        assert len(scales) == 1, f"Expected 1 deduplicated scale, got {len(scales)}: {scales}"
        assert scales[0] == 10.0
    finally:
        ss._score_scale_candidate = original_coarse
        ss.fine_scale_search = original_fine
        ss.ultra_fine_scale_search = original_ultra

if __name__ == '__main__':
    pytest.main(['-v', 'test_dedup.py'])
