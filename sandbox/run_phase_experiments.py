import cv2
import numpy as np
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.config import Config
from src.pipeline_factory import build_default_pipeline

def fine_grain_sweep():
    # Sweep crop from 0.15 to 0.40 with step 0.01
    crops = np.arange(0.15, 0.41, 0.01)
    
    print("=== Fine-grained Crop Sweep with Blur=9, Canny=(30/100) ===")
    print(f"{'Crop':<6} | {'Near Score':<10} | {'Far Score':<10} | {'Avg Score':<10}")
    print("-" * 45)
    
    best_config = None
    best_avg = 0
    
    for crop in crops:
        crop = round(float(crop), 2)
        config = Config(
            canny_threshold1=30.0,
            canny_threshold2=100.0,
            gaussian_blur_kernel=9,
            photo_bottom_crop_ratio=crop
        )
        pipeline = build_default_pipeline(config)
        
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
        
        pairs = pipeline.load_directory("imgs/origin", "imgs/dummy")
        scores = {}
        for origin_file, dummy_file, origin, dummy in pairs:
            if "実写" in dummy_file:
                result = pipeline.process_pair(origin, dummy, method="diff")
                scores[dummy_file] = result.similarity_score
        
        if len(scores) < 2:
            continue
            
        near = scores["実写直近(ドトール).jpg"]
        far = scores["実写遠近(ドトール).jpg"]
        avg = (near + far) / 2
        
        print(f"{crop:<6.2f} | {near:<10.4f} | {far:<10.4f} | {avg:<10.4f}")
        
        if avg > best_avg:
            best_avg = avg
            best_config = (crop, near, far)
            
    print("-" * 45)
    print(f"Best Configuration: Crop={best_config[0]:.2f} (Near={best_config[1]:.4f}, Far={best_config[2]:.4f}, Avg={best_avg:.4f})")

if __name__ == "__main__":
    fine_grain_sweep()
