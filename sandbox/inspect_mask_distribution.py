import cv2
import numpy as np
import os
import sys
from pathlib import Path

def analyze_mask():
    ROOT = Path("/Users/abekoudai/Desktop/test_ground")
    sys.path.insert(0, str(ROOT))
    
    from src.config import Config
    from src.pipeline_factory import build_default_pipeline
    
    config = Config(photo_bottom_crop_ratio=0.26)
    pipeline = build_default_pipeline(config)
    
    img = cv2.imread(str(ROOT / "imgs/dummy/実写遠近(ドトール).jpg"))
    cropped = pipeline.normalizer._crop_letter_roi(img)
    mask = pipeline.normalizer._extract_letter_mask(cropped)
    
    h, w = mask.shape
    print(f"Image height: {h}, width: {w}")
    
    # Check bottom lines
    print("Mask pixel counts in the bottom 50 rows:")
    for y in range(h - 10, h):
        row_sum = np.sum(mask[y, :] > 0)
        print(f"  Row y={y}: {row_sum} pixels")
        
    # Find contours that touch the bottom
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    touching_bottom = 0
    for idx, c in enumerate(contours):
        ys = c[:, 0, 1]
        max_y = ys.max()
        area = cv2.contourArea(c)
        if max_y >= h - 5:
            touching_bottom += 1
            print(f"  Contour {idx} touches bottom! Area={area:.1f}, MaxY={max_y}")
            # Print bounding box
            bx, by, bw, bh = cv2.boundingRect(c)
            print(f"    Bounding Box: x={bx}, y={by}, w={bw}, h={bh}")
            
if __name__ == "__main__":
    analyze_mask()
