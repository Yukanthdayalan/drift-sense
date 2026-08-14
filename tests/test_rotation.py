import os
import math
import numpy as np
import cv2
import pytest

from drift_sense.dataset import GeneratorConfig, generate_sample
from drift_sense.matcher import match

def create_sample(angle, tmp_path):
    config = GeneratorConfig()
    config.rotation_range = 0.0 # Force no random rotation
    
    # We will generate a base sample, then manually rotate it to exact angle
    ref, search, meta = generate_sample(1234, config)
    
    if angle != 0.0:
        h, w = ref.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        ref = cv2.warpAffine(ref, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
    ref_path = str(tmp_path / f"ref_{angle}.png")
    search_path = str(tmp_path / f"search_{angle}.png")
    
    cv2.imwrite(ref_path, ref)
    cv2.imwrite(search_path, search)
    
    return ref_path, search_path, meta

@pytest.mark.parametrize("angle", [0.0, 7.5, -7.5, 15.0, -15.0])
def test_rotation_angles(angle, tmp_path):
    ref_path, search_path, meta = create_sample(angle, tmp_path)
    
    res = match(ref_path, search_path)
    
    err = math.sqrt((res.prediction.x - meta.gt_center_x)**2 + (res.prediction.y - meta.gt_center_y)**2)
    assert err < 2.0, f"Error {err:.2f} px exceeds 2.0 px limit at {angle} degrees"
