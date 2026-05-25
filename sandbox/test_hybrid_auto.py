import cv2
import numpy as np
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.pipeline_factory import build_default_pipeline

def calculate_iou(img1, img2):
    def get_dark_mask(img):
        if len(img.shape) == 2:
            return (img < 120).astype(np.uint8) * 255
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        return (v < 120).astype(np.uint8) * 255
        
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h))
        
    bin1 = get_dark_mask(img1)
    bin2 = get_dark_mask(img2)
    
    intersection = cv2.bitwise_and(bin1, bin2)
    union = cv2.bitwise_or(bin1, bin2)
    
    num_inter = np.sum(intersection > 0)
    num_union = np.sum(union > 0)
    
    if num_union == 0:
        return 0.0
    return float(num_inter) / float(num_union)

def test_hybrid_auto():
    config = Config(
        canny_threshold1=30.0,
        canny_threshold2=100.0,
        gaussian_blur_kernel=7,
        photo_bottom_crop_ratio=0.26,
        hu_moments_compare_limit=7
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

    def get_warped_and_flat(cropped, src_pts, dst_pts, ow, oh):
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(cropped, M, (ow, oh), borderValue=(255, 255, 255))
        mask = pipeline.normalizer._blue_mask_hsv(warped)
        flat = pipeline.normalizer._mask_to_flat_cad(mask)
        return flat

    def evaluate_methods(photo_bgr, origin_bgr):
        cropped = pipeline.normalizer._crop_letter_roi(photo_bgr)
        mask = pipeline.normalizer._extract_letter_mask(cropped)
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
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_pts = [c for c in contours if cv2.contourArea(c) > 10]
        if not all_pts:
            x0, y0, x1, y1 = pipeline.normalizer._auto_crop_to_mask(mask)
            flat = pipeline.normalizer._mask_to_flat_cad(mask[y0:y1, x0:x1])
            resized = cv2.resize(flat, (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_AREA)
            full = np.full((oh, ow, 3), (255, 255, 255), dtype=np.uint8)
            full[oy0:oy1, ox0:ox1] = resized
            return {"fallback": full}
            
        merged_pts = np.vstack(all_pts)
        hull = cv2.convexHull(merged_pts)
        
        # 1. RotatedRect
        rect = cv2.minAreaRect(hull)
        corners_rotated = pipeline.normalizer._order_points(np.array(cv2.boxPoints(rect), dtype=np.float32))
        
        # Destination mappings
        dst_stretch = np.array([
            [ox0, oy0],
            [ox1, oy0],
            [ox1, oy1],
            [ox0, oy1]
        ], dtype=np.float32)
        
        methods = {
            "rotated": get_warped_and_flat(cropped, corners_rotated, dst_stretch, ow, oh),
        }
        return methods

    real_images = [
        ("実写直近(ドトール).jpg", "imgs/dummy/実写直近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png"),
        ("実写遠近(ドトール).jpg", "imgs/dummy/実写遠近(ドトール).jpg", "imgs/origin/POLITEC(ドトール).png"),
        ("dummy(ドトール)入れ替えのみ_00_TEOPCIL.jpg", "imgs/dummy/dummy(ドトール)入れ替えのみ_00_TEOPCIL.jpg", "imgs/origin/POLITEC(ドトール).png"),
        ("dummy(吉野家)ローマ字入れ替えズレあり_00_NIUAAOK.jpg", "imgs/dummy/dummy(吉野家)ローマ字入れ替えズレあり_00_NIUAAOK.jpg", "imgs/origin/能開大(吉野家).png"),
        ("dummy(吉野家)入れ替えズレあり_00_大能開.jpg", "imgs/dummy/dummy(吉野家)入れ替えズレあり_00_大能開.jpg", "imgs/origin/能開大(吉野家).png"),
        ("dummy(吉野家)入れ替えズレあり_01_能開大.jpg", "imgs/dummy/dummy(吉野家)入れ替えズレあり_01_能開大.jpg", "imgs/origin/能開大(吉野家).png")
    ]
    
    out_dir = ROOT / "sandbox" / "output"
    out_dir.mkdir(exist_ok=True)
    
    print("=== Multi-Configuration Normalizer, Hu and Brightness-IoU Scorer Comparison ===")
    for filename, dummy_path, origin_path in real_images:
        origin = cv2.imread(str(ROOT / origin_path))
        dummy = cv2.imread(str(ROOT / dummy_path))
        
        # Check if needs normalization (same as pipeline)
        needs_norm = max(dummy.shape[:2]) > max(origin.shape[:2]) * config.photo_size_ratio_threshold
        
        if needs_norm:
            methods = evaluate_methods(dummy, origin)
            flat_img = methods["rotated"]
        else:
            flat_img = dummy
        
        # IoU score (using new brightness-based calculate_iou)
        score_iou = calculate_iou(origin, flat_img)
        print(f"  * {filename:50}: IoU (Overlap)={score_iou:.4f} | NeedsNorm={needs_norm}")
        
        # Save output image
        res = pipeline.matcher.match_shapes(origin, flat_img, method="diff")
        res.photo_normalized = needs_norm
        res.chosen_method = "rotated" if needs_norm else "none"
        res.similarity_score = score_iou
        pipeline.exporter.export_image(res, str(out_dir / f"{filename}_iou_debug.png"))

if __name__ == "__main__":
    test_hybrid_auto()
