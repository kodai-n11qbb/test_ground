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
        # Save output image
        out_path = ROOT / "sandbox" / "output" / f"{filename}_test_clean_mapping.png"
        pipeline.exporter.export_image(result, str(out_path))

if __name__ == "__main__":
    test_clean_far()
