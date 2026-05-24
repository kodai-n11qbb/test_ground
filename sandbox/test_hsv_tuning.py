import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

ROOT = Path("/Users/abekoudai/Desktop/test_ground")
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.pipeline_factory import build_default_pipeline

def parse_csv_ints(s):
    return [int(x) for x in s.split(',')]

def run_tuning_pipeline(hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2, threshold):
    config = Config(
        canny_threshold1=30.0,
        canny_threshold2=100.0,
        gaussian_blur_kernel=7,
        photo_bottom_crop_ratio=0.26,
        photo_corner_detection_method="rotated",
        photo_alignment_mode="stretch",
        match_threshold=threshold,
        match_method="iou"
    )
    
    pipeline = build_default_pipeline(config)
    
    # Helper to warp
    def get_warped_and_flat(cropped, src_pts, dst_pts, ow, oh):
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(cropped, M, (ow, oh), borderValue=(255, 255, 255))
        mask = custom_blue_mask_hsv(warped)
        flat = pipeline.normalizer._mask_to_flat_cad(mask)
        return flat

    # Overwrite _blue_mask_hsv in normalizer
    def custom_blue_mask_hsv(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lower1, dtype=np.uint8), np.array(hsv_upper1, dtype=np.uint8))
        mask2 = cv2.inRange(hsv, np.array(hsv_lower2, dtype=np.uint8), np.array(hsv_upper2, dtype=np.uint8))
        return cv2.bitwise_or(mask, mask2)
        
    pipeline.normalizer._blue_mask_hsv = custom_blue_mask_hsv
    
    # Overwrite _compare_iou in matcher to use this customized blue mask
    def custom_compare_iou(origin_img, dummy_img):
        h, w = origin_img.shape[:2]
        dummy_resized = dummy_img
        if dummy_img.shape[:2] != (h, w):
            dummy_resized = cv2.resize(dummy_img, (w, h))
            
        bin_orig = custom_blue_mask_hsv(origin_img)
        bin_dum = custom_blue_mask_hsv(dummy_resized)
        
        intersection = cv2.bitwise_and(bin_orig, bin_dum)
        union = cv2.bitwise_or(bin_orig, bin_dum)
        
        num_inter = np.sum(intersection > 0)
        num_union = np.sum(union > 0)
        
        if num_union == 0:
            return 0.0
        return float(num_inter) / float(num_union)
        
    pipeline.matcher._compare_iou = custom_compare_iou
    
    # Overwrite match_shapes logic to route to custom iou
    def custom_match_shapes(origin_img, dummy_img, method=None):
        origin_processed = pipeline.matcher._preprocess(origin_img)
        dummy_processed = pipeline.matcher._preprocess(dummy_img)
        origin_contours = pipeline.matcher._extract_contours(origin_processed)
        dummy_contours = pipeline.matcher._extract_contours(dummy_processed)
        
        similarity = custom_compare_iou(origin_img, dummy_img)
        is_match = similarity >= config.match_threshold
        
        h, w = origin_img.shape[:2]
        dummy_resized = dummy_img
        if dummy_img.shape[:2] != (h, w):
            dummy_resized = cv2.resize(dummy_img, (w, h))
        diff = cv2.absdiff(origin_img, dummy_resized)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        from src.models import MatchResult
        return MatchResult(
            similarity_score=similarity,
            is_match=is_match,
            origin_path="",
            dummy_path="",
            hu_moments_origin=np.zeros(7),
            hu_moments_dummy=np.zeros(7),
            diff_mask=diff_mask,
            origin_img=origin_img,
            dummy_img=dummy_resized,
            origin_contours=origin_contours,
            dummy_contours=dummy_contours
        )
        
    pipeline.matcher.match_shapes = custom_match_shapes

    real_images = [
        ("実写直近(ドトール).jpg", "imgs/dummy/実写直近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png"),
        ("実写遠近(ドトール).jpg", "imgs/dummy/実写遠近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png"),
        ("dummy(ドトール)入れ替えのみ_00_TEOPCIL.jpg", "imgs/dummy/dummy(ドトール)入れ替えのみ_00_TEOPCIL.jpg", "imgs/origin/POLITEC(ドトール).png")
    ]
    
    out_dir = ROOT / "sandbox" / "output"
    out_dir.mkdir(exist_ok=True)
    
    results = {}
    print("=== Tuning Pipeline Run ===")
    for filename, dummy_path, origin_path in real_images:
        origin = cv2.imread(str(ROOT / origin_path))
        dummy = cv2.imread(str(ROOT / dummy_path))
        
        needs_norm = max(dummy.shape[:2]) > max(origin.shape[:2]) * config.photo_size_ratio_threshold
        
        if needs_norm:
            cropped = pipeline.normalizer._crop_letter_roi(dummy)
            mask = pipeline.normalizer._extract_letter_mask(cropped)
            oh, ow = origin.shape[:2]
            
            origin_gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
            _, origin_mask = cv2.threshold(origin_gray, 200, 255, cv2.THRESH_BINARY_INV)
            ys_orig, xs_orig = np.where(origin_mask > 0)
            if len(xs_orig) > 0:
                ox0, ox1 = int(xs_orig.min()), int(xs_orig.max())
                oy0, oy1 = int(ys_orig.min()), int(ys_orig.max())
            else:
                ox0, oy0, ox1, oy1 = 0, 0, ow - 1, oh - 1
                
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_pts = [c for c in contours if cv2.contourArea(c) > 10]
            if not all_pts:
                flat_img = dummy
            else:
                merged_pts = np.vstack(all_pts)
                hull = cv2.convexHull(merged_pts)
                rect = cv2.minAreaRect(hull)
                corners_rotated = pipeline.normalizer._order_points(np.array(cv2.boxPoints(rect), dtype=np.float32))
                
                dst_stretch = np.array([
                    [ox0, oy0],
                    [ox1, oy0],
                    [ox1, oy1],
                    [ox0, oy1]
                ], dtype=np.float32)
                
                flat_img = get_warped_and_flat(cropped, corners_rotated, dst_stretch, ow, oh)
        else:
            flat_img = dummy
            
        res = pipeline.matcher.match_shapes(origin, flat_img)
        res.photo_normalized = needs_norm
        res.chosen_method = "rotated"
        
        print(f"  {filename}: IoU={res.similarity_score:.4f} (Match: {res.is_match})")
        results[filename] = {"score": res.similarity_score, "is_match": res.is_match}
        
        out_name = filename.replace('.', '_')
        cv2.imwrite(str(out_dir / f"{out_name}_flat.png"), flat_img)
        pipeline.exporter.export_image(res, str(out_dir / f"{out_name}_overlay.png"))
        
        # Save individual mask for debugging HSV mask in UI
        cv2.imwrite(str(out_dir / f"{out_name}_hsv_mask.png"), custom_blue_mask_hsv(flat_img))
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lower1', type=str, default='90,40,30')
    parser.add_argument('--upper1', type=str, default='140,255,255')
    parser.add_argument('--lower2', type=str, default='100,20,20')
    parser.add_argument('--upper2', type=str, default='150,255,180')
    parser.add_argument('--threshold', type=float, default=0.70)
    args = parser.parse_args()
    
    hsv_lower1 = parse_csv_ints(args.lower1)
    hsv_upper1 = parse_csv_ints(args.upper1)
    hsv_lower2 = parse_csv_ints(args.lower2)
    hsv_upper2 = parse_csv_ints(args.upper2)
    
    run_tuning_pipeline(hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2, args.threshold)

if __name__ == '__main__':
    main()
