import cv2
import numpy as np
import os
import sys
from pathlib import Path

def test_clean_far():
    ROOT = Path("/Users/abekoudai/Desktop/test_ground")
    sys.path.insert(0, str(ROOT))
    
    from src.config import Config
    from src.pipeline_factory import build_default_pipeline
    
    # Use optimized config with default mapping to full image boundaries
    config = Config(
        canny_threshold1=30.0,
        canny_threshold2=100.0,
        gaussian_blur_kernel=7,
        photo_bottom_crop_ratio=0.26
    )
    pipeline = build_default_pipeline(config)
    
    # Custom band removal to remove bottom table noise (bh < h * 0.15)
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
    
    # Custom Log-scaled Hu moments comparison using first 2 moments
    def custom_compare_hu_moments(hu1, hu2):
        if hu1 is None or hu2 is None:
            return 0.0
        hu1 = hu1[:2]
        hu2 = hu2[:2]
        hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-15)
        hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-15)
        diff = np.abs(hu1_log - hu2_log)
        mean_diff = np.mean(diff)
        return 1.0 / (1.0 + 0.1 * mean_diff)
    pipeline.matcher._compare_hu_moments = custom_compare_hu_moments

    # Custom corner detection using direct hull extremum points
    def custom_detect_four_corners(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        all_points = []
        for c in contours:
            if cv2.contourArea(c) > 10:
                all_points.append(c)
                
        if not all_points:
            return None
            
        merged_points = np.vstack(all_points)
        hull = cv2.convexHull(merged_points).reshape(-1, 2)
        
        # Select 4 extremum points
        # Top-Left: minimize x + y
        # Bottom-Right: maximize x + y
        # Top-Right: maximize x - y
        # Bottom-Left: minimize x - y
        pts = hull.astype(np.float32)
        s = pts.sum(axis=1)
        diff = pts[:, 0] - pts[:, 1]
        
        tl = hull[np.argmin(s)]
        tr = hull[np.argmax(diff)]
        br = hull[np.argmax(s)]
        bl = hull[np.argmin(diff)]
        
        corners = np.array([tl, tr, br, bl], dtype=np.int32)
        print(f"    [DEBUG] custom_detect_four_corners returned: {corners.tolist()}")
        return corners
    pipeline.normalizer._detect_four_corners = custom_detect_four_corners

    # Wrap normalize to export intermediate images
    orig_normalize = pipeline.normalizer.normalize
    def debug_normalize(photo_bgr, origin_bgr):
        cropped = pipeline.normalizer._crop_letter_roi(photo_bgr)
        mask = pipeline.normalizer._extract_letter_mask(cropped)
        corners = pipeline.normalizer._detect_four_corners(mask)
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
            
            # Target bounding box dimensions
            ox_w = ox1 - ox0
            ox_h = oy1 - oy0
            dst_ar = ox_w / ox_h
            
            if src_ar > dst_ar:
                # Fit horizontally, center vertically within origin bounding box
                new_h = ox_w / src_ar
                oy_offset = (ox_h - new_h) / 2.0
                dst_pts = np.array([
                    [ox0, oy0 + oy_offset],
                    [ox1, oy0 + oy_offset],
                    [ox1, oy0 + oy_offset + new_h],
                    [ox0, oy0 + oy_offset + new_h]
                ], dtype=np.float32)
            else:
                # Fit vertically, center horizontally within origin bounding box
                new_w = ox_h * src_ar
                ox_offset = (ox_w - new_w) / 2.0
                dst_pts = np.array([
                    [ox0 + ox_offset, oy0],
                    [ox0 + ox_offset + new_w, oy0],
                    [ox0 + ox_offset + new_w, oy1],
                    [ox0 + ox_offset, oy1]
                ], dtype=np.float32)
                
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            cv2.imwrite(str(ROOT / "sandbox" / "output" / "debug_input_mask.png"), mask)
            warped = cv2.warpPerspective(cropped, M, (ow, oh), borderValue=(255, 255, 255))
            cv2.imwrite(str(ROOT / "sandbox" / "output" / "debug_warped.png"), warped)
            
            warped_mask = pipeline.normalizer._blue_mask_hsv(warped)
            cv2.imwrite(str(ROOT / "sandbox" / "output" / "debug_warped_mask.png"), warped_mask)
            
            flat = pipeline.normalizer._mask_to_flat_cad(warped_mask)
            cv2.imwrite(str(ROOT / "sandbox" / "output" / "debug_flat.png"), flat)
            return flat
        else:
            return orig_normalize(photo_bgr, origin_bgr)
            
    pipeline.normalizer.normalize = debug_normalize

    real_images = [
        ("実写直近(ドトール).jpg", "imgs/dummy/実写直近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png"),
        ("実写遠近(ドトール).jpg", "imgs/dummy/実写遠近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png")
    ]
    
    print("=== Test: Cleaned Band + Original Full-Boundary Mapping ===")
    for filename, dummy_path, origin_path in real_images:
        origin = cv2.imread(str(ROOT / origin_path))
        dummy = cv2.imread(str(ROOT / dummy_path))
        result = pipeline.process_pair(origin, dummy, method="diff")
        print(f"  {filename}: Score={result.similarity_score:.4f}")
        print(f"    Origin Hu: {result.hu_moments_origin.tolist() if result.hu_moments_origin is not None else None}")
        print(f"    Dummy Hu:  {result.hu_moments_dummy.tolist() if result.hu_moments_dummy is not None else None}")
        # Save output image
        out_path = ROOT / "sandbox" / "output" / f"{filename}_test_clean_mapping.png"
        pipeline.exporter.export_image(result, str(out_path))

if __name__ == "__main__":
    test_clean_far()
