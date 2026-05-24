import cv2
import numpy as np
import os
import sys
from pathlib import Path

def test_ar_preservation():
    ROOT = Path("/Users/abekoudai/Desktop/test_ground")
    sys.path.insert(0, str(ROOT))
    
    from src.config import Config
    from src.pipeline_factory import build_default_pipeline
    
    config = Config(
        canny_threshold1=30.0,
        canny_threshold2=100.0,
        gaussian_blur_kernel=7,
        photo_bottom_crop_ratio=0.26
    )
    pipeline = build_default_pipeline(config)
    
    # Custom band removal
    def custom_remove_bottom_band_artifacts(mask):
        h, w = mask.shape[:2]
        y_line = int(h * (1.0 - 0.15))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = mask.copy()
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if y + bh < y_line:
                continue
            if bw > w * 0.25 and bh < h * 0.15:
                out[labels == i] = 0
        return out
    pipeline.normalizer._remove_bottom_band_artifacts = custom_remove_bottom_band_artifacts
    
    # Log-scaled Hu moments comparison
    def custom_compare_hu_moments(hu1, hu2):
        if hu1 is None or hu2 is None:
            return 0.0
        hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-15)
        hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-15)
        diff = np.abs(hu1_log - hu2_log)
        mean_diff = np.mean(diff)
        return 1.0 / (1.0 + 0.1 * mean_diff)
    pipeline.matcher._compare_hu_moments = custom_compare_hu_moments
    
    # Custom normalize with Aspect Ratio Preservation
    def custom_normalize(photo_bgr, origin_bgr):
        cropped = pipeline.normalizer._crop_letter_roi(photo_bgr)
        mask = pipeline.normalizer._extract_letter_mask(cropped)
        
        corners = pipeline.normalizer._detect_four_corners(mask)
        oh, ow = origin_bgr.shape[:2]
        
        if corners is not None:
            src_pts = pipeline.normalizer._order_points(corners)
            
            # Calculate source quad average width and height
            w_top = np.linalg.norm(src_pts[1] - src_pts[0])
            w_bot = np.linalg.norm(src_pts[2] - src_pts[3])
            h_left = np.linalg.norm(src_pts[3] - src_pts[0])
            h_right = np.linalg.norm(src_pts[2] - src_pts[1])
            
            w_avg = (w_top + w_bot) / 2.0
            h_avg = (h_left + h_right) / 2.0
            src_ar = w_avg / h_avg if h_avg > 0 else 1.0
            
            # Canvas aspect ratio
            canvas_ar = ow / oh
            
            if src_ar > canvas_ar:
                # Fit horizontally, center vertically
                dst_w = ow
                dst_h = ow / src_ar
                oy = (oh - dst_h) / 2.0
                dst_pts = np.array([
                    [0, oy],
                    [ow - 1, oy],
                    [ow - 1, oy + dst_h],
                    [0, oy + dst_h]
                ], dtype=np.float32)
            else:
                # Fit vertically, center horizontally
                dst_h = oh
                dst_w = oh * src_ar
                ox = (ow - dst_w) / 2.0
                dst_pts = np.array([
                    [ox, 0],
                    [ox + dst_w, 0],
                    [ox + dst_w, oh - 1],
                    [ox, oh - 1]
                ], dtype=np.float32)
                
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(cropped, M, (ow, oh), borderValue=(255, 255, 255))
            
            warped_mask = pipeline.normalizer._extract_letter_mask(warped)
            flat = pipeline.normalizer._mask_to_flat_cad(warped_mask)
            return flat
        else:
            # Fallback
            x0, y0, x1, y1 = pipeline.normalizer._auto_crop_to_mask(mask)
            flat = pipeline.normalizer._mask_to_flat_cad(mask[y0:y1, x0:x1])
            return cv2.resize(flat, (ow, oh), interpolation=cv2.INTER_AREA)
            
    pipeline.normalizer.normalize = custom_normalize

    real_images = [
        ("実写直近(ドトール).jpg", "imgs/dummy/実写直近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png"),
        ("実写遠近(ドトール).jpg", "imgs/dummy/実写遠近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png")
    ]
    
    print("=== Aspect Ratio Preserving Perspective Warp Test ===")
    for filename, dummy_path, origin_path in real_images:
        origin = cv2.imread(str(ROOT / origin_path))
        dummy = cv2.imread(str(ROOT / dummy_path))
        result = pipeline.process_pair(origin, dummy, method="diff")
        print(f"  {filename}: Score={result.similarity_score:.4f}")
        # Save output image
        out_path = ROOT / "sandbox" / "output" / f"{filename}_test_ar_preserve.png"
        pipeline.exporter.export_image(result, str(out_path))

if __name__ == "__main__":
    test_ar_preservation()
