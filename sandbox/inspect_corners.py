import cv2
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, "/Users/abekoudai/Desktop/test_ground")

from src.config import Config
from src.pipeline_factory import build_default_pipeline

def inspect_corners():
    config = Config(photo_bottom_crop_ratio=0.26)
    pipeline = build_default_pipeline(config)
    
    real_images = [
        ("実写直近(ドトール).jpg", "imgs/dummy/実写直近(ドトール).jpg"),
        ("実写遠近(ドトール).jpg", "imgs/dummy/実写遠近(ドトール).jpg")
    ]
    
    for filename, path in real_images:
        print(f"\n=== Inspecting Corners for: {filename} ===")
        img = cv2.imread(str(ROOT / path))
        cropped = pipeline.normalizer._crop_letter_roi(img)
        mask = pipeline.normalizer._extract_letter_mask(cropped)
        
        # Get dimensions
        ch, cw = cropped.shape[:2]
        print(f"  Cropped Image Size: {cw}x{ch}")
        
        # Analyze contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  Total contours found: {len(contours)}")
        
        large_contours = [c for c in contours if cv2.contourArea(c) > 10]
        print(f"  Contours with area > 10: {len(large_contours)}")
        
        if large_contours:
            areas = [cv2.contourArea(c) for c in large_contours]
            print(f"    Min area: {min(areas):.1f}, Max area: {max(areas):.1f}, Mean area: {np.mean(areas):.1f}")
            
            # Merged hull
            merged = np.vstack(large_contours)
            hull = cv2.convexHull(merged)
            print(f"    Merged convex hull vertices count: {len(hull)}")
            
            # approxPolyDP sweep
            peri = cv2.arcLength(hull, True)
            detected = None
            for eps in np.linspace(0.01, 0.2, 50):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    detected = approx.reshape(4, 2)
                    print(f"    approxPolyDP succeeded at eps={eps:.4f} -> Corners: {detected.tolist()}")
                    break
                    
            if detected is None:
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                print(f"    approxPolyDP failed. Using minAreaRect -> Corners: {box.tolist()}")
                detected = np.array(box, dtype=np.int32)
                
            # Print ordered points
            ordered = pipeline.normalizer._order_points(detected)
            print(f"    Ordered Corners: {ordered.tolist()}")
            
            # Check aspect ratio of the detected quad
            # Widths (top, bottom) and Heights (left, right)
            w_top = np.linalg.norm(ordered[1] - ordered[0])
            w_bot = np.linalg.norm(ordered[2] - ordered[3])
            h_left = np.linalg.norm(ordered[3] - ordered[0])
            h_right = np.linalg.norm(ordered[2] - ordered[1])
            print(f"    Measured Widths: Top={w_top:.1f}, Bottom={w_bot:.1f}")
            print(f"    Measured Heights: Left={h_left:.1f}, Right={h_right:.1f}")
            aspect_ratio = ((w_top + w_bot) / 2) / ((h_left + h_right) / 2)
            print(f"    Detected Aspect Ratio (W/H): {aspect_ratio:.3f}")
            
if __name__ == "__main__":
    ROOT = Path("/Users/abekoudai/Desktop/test_ground")
    inspect_corners()
