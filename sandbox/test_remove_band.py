import cv2
import numpy as np
import os
import sys
from pathlib import Path

def test_removal():
    ROOT = Path("/Users/abekoudai/Desktop/test_ground")
    sys.path.insert(0, str(ROOT))
    
    from src.config import Config
    from src.pipeline_factory import build_default_pipeline
    
    config = Config(photo_bottom_crop_ratio=0.26)
    pipeline = build_default_pipeline(config)
    
    # Override _remove_bottom_band_artifacts with looser height threshold
    def custom_remove_bottom_band_artifacts(mask):
        h, w = mask.shape[:2]
        y_line = int(h * (1.0 - 0.15)) # Look at bottom 15%
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = mask.copy()
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if y + bh < y_line:
                continue
            # Increase height limit to 15% of image height
            if bw > w * 0.25 and bh < h * 0.15:
                print(f"  [Removal Test] Removing component {i}: x={x}, y={y}, w={bw}, h={bh}, area={area}")
                out[labels == i] = 0
        return out
        
    pipeline.normalizer._remove_bottom_band_artifacts = custom_remove_bottom_band_artifacts
    
    img = cv2.imread(str(ROOT / "imgs/dummy/実写遠近(ドトール).jpg"))
    ref = cv2.imread(str(ROOT / "imgs/origin/POLITEC(ドトール).png"))
    
    # Process
    cropped = pipeline.normalizer._crop_letter_roi(img)
    mask = pipeline.normalizer._extract_letter_mask(cropped)
    
    # Check corners now
    corners = pipeline.normalizer._detect_four_corners(mask)
    if corners is not None:
        pts = pipeline.normalizer._order_points(corners)
        print(f"  New Corners detected: {pts.tolist()}")
        w_top = np.linalg.norm(pts[1] - pts[0])
        w_bot = np.linalg.norm(pts[2] - pts[3])
        h_left = np.linalg.norm(pts[3] - pts[0])
        h_right = np.linalg.norm(pts[2] - pts[1])
        aspect_ratio = ((w_top + w_bot) / 2) / ((h_left + h_right) / 2)
        print(f"  New Aspect Ratio: {aspect_ratio:.3f}")
        
        # Calculate new similarity score (using Log-scaled Hu moments)
        def custom_compare_hu_moments(hu1, hu2):
            if hu1 is None or hu2 is None:
                return 0.0
            hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-15)
            hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-15)
            diff = np.abs(hu1_log - hu2_log)
            mean_diff = np.mean(diff)
            return 1.0 / (1.0 + 0.1 * mean_diff)
        pipeline.matcher._compare_hu_moments = custom_compare_hu_moments
        
        result = pipeline.process_pair(ref, img, method="diff")
        print(f"  New Similarity Score: {result.similarity_score:.4f}")
        
if __name__ == "__main__":
    test_removal()
