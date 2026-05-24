import cv2
import numpy as np
import os
import sys
from pathlib import Path

def test_rotated_rect_align():
    ROOT = Path("/Users/abekoudai/Desktop/test_ground")
    sys.path.insert(0, str(ROOT))
    
    from src.config import Config
    from src.pipeline_factory import build_default_pipeline
    
    # Use optimized config
    config = Config(
        canny_threshold1=30.0,
        canny_threshold2=100.0,
        gaussian_blur_kernel=9,
        photo_bottom_crop_ratio=0.26
    )
    pipeline = build_default_pipeline(config)
    
    # Custom band removal to remove bottom table noise
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
    
    # Custom Log-scaled Hu moments comparison
    def custom_compare_hu_moments(hu1, hu2):
        if hu1 is None or hu2 is None:
            return 0.0
        hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-15)
        hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-15)
        diff = np.abs(hu1_log - hu2_log)
        mean_diff = np.mean(diff)
        return 1.0 / (1.0 + 0.1 * mean_diff)
    pipeline.matcher._compare_hu_moments = custom_compare_hu_moments
    
    # Custom normalize using RotatedRect corners
    def custom_normalize(photo_bgr, origin_bgr):
        cropped = pipeline.normalizer._crop_letter_roi(photo_bgr)
        mask = pipeline.normalizer._extract_letter_mask(cropped)
        
        # Get convex hull of the mask contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        oh, ow = origin_bgr.shape[:2]
        
        # Get origin letter bounding box
        origin_gray = cv2.cvtColor(origin_bgr, cv2.COLOR_BGR2GRAY)
        _, origin_mask = cv2.threshold(origin_gray, 200, 255, cv2.THRESH_BINARY_INV)
        ys_orig, xs_orig = np.where(origin_mask > 0)
        if len(xs_orig) > 0:
            ox0, ox1 = int(xs_orig.min()), int(xs_orig.max())
            oy0, oy1 = int(ys_orig.min()), int(ys_orig.max())
        else:
            ox0, oy0, ox1, oy1 = 0, 0, ow - 1, oh - 1
            
        all_pts = []
        for c in contours:
            if cv2.contourArea(c) > 10:
                all_pts.append(c)
                
        if all_pts:
            merged_pts = np.vstack(all_pts)
            hull = cv2.convexHull(merged_pts)
            
            # Use RotatedRect instead of approxPolyDP!
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            corners = np.array(box, dtype=np.float32)
            
            src_pts = pipeline.normalizer._order_points(corners)
            dst_pts = np.array([
                [ox0, oy0],
                [ox1, oy0],
                [ox1, oy1],
                [ox0, oy1]
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
            resized = cv2.resize(flat, (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_AREA)
            full = np.full((oh, ow, 3), (255, 255, 255), dtype=np.uint8)
            full[oy0:oy1, ox0:ox1] = resized
            return full
            
    pipeline.normalizer.normalize = custom_normalize

    real_images = [
        ("実写直近(ドトール).jpg", "imgs/dummy/実写直近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png"),
        ("実写遠近(ドトール).jpg", "imgs/dummy/実写遠近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png")
    ]
    
    print("=== Alignment Test results (Using RotatedRect Corners mapped to Origin Box) ===")
    for filename, dummy_path, origin_path in real_images:
        origin = cv2.imread(str(ROOT / origin_path))
        dummy = cv2.imread(str(ROOT / dummy_path))
        result = pipeline.process_pair(origin, dummy, method="diff")
        print(f"  {filename}: Score={result.similarity_score:.4f}")
        # Save output image
        out_path = ROOT / "sandbox" / "output" / f"{filename}_test_rotated.png"
        pipeline.exporter.export_image(result, str(out_path))

if __name__ == "__main__":
    test_rotated_rect_align()
