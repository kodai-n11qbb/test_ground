import cv2
import numpy as np
import os
import sys
from pathlib import Path

def compare_boxes():
    ROOT = Path("/Users/abekoudai/Desktop/test_ground")
    sys.path.insert(0, str(ROOT))
    
    from src.config import Config
    from src.pipeline_factory import build_default_pipeline
    
    config = Config()
    pipeline = build_default_pipeline(config)
    
    img = cv2.imread(str(ROOT / "imgs/dummy/実写直近(ドトール).jpg"))
    cropped = pipeline.normalizer._crop_letter_roi(img)
    mask = pipeline.normalizer._extract_letter_mask(cropped)
    
    # 1. Tight bounding box of mask pixels
    ys, xs = np.where(mask > 0)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    print(f"Mask Tight Bounding Box: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
    print(f"  Width={x1-x0}, Height={y1-y0}, Aspect Ratio={(x1-x0)/(y1-y0):.3f}")
    
    # 2. Detected corners
    corners = pipeline.normalizer._detect_four_corners(mask)
    if corners is not None:
        pts = pipeline.normalizer._order_points(corners)
        print(f"Detected Corners:")
        for idx, pt in enumerate(pts):
            print(f"  Point {idx}: {pt.tolist()}")
            
if __name__ == "__main__":
    compare_boxes()
