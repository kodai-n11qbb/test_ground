import cv2
import numpy as np
import os
import sys
import json
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.pipeline_factory import build_default_pipeline

def run_experiment_with_config(c1, c2, blur, log_hu=True):
    config = Config(
        canny_threshold1=c1,
        canny_threshold2=c2,
        gaussian_blur_kernel=blur
    )
    pipeline = build_default_pipeline(config)
    
    if log_hu:
        def custom_compare_hu_moments(hu1, hu2):
            if hu1 is None or hu2 is None:
                return 0.0
            hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-15)
            hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-15)
            diff = np.abs(hu1_log - hu2_log)
            mean_diff = np.mean(diff)
            return 1.0 / (1.0 + 0.1 * mean_diff)
        pipeline.matcher._compare_hu_moments = custom_compare_hu_moments

    pairs = pipeline.load_directory(str(ROOT / "imgs" / "origin"), str(ROOT / "imgs" / "dummy"))
    
    results = {}
    for origin_file, dummy_file, origin, dummy in pairs:
        if "ドトール" in dummy_file:
            result = pipeline.process_pair(origin, dummy, method="diff")
            results[dummy_file] = float(result.similarity_score)
            
    return results

def main():
    print("Starting Parameter Sweep in Sandbox...")
    
    blurs = [3, 5, 7, 9]
    canny_pairs = [
        (30.0, 100.0),
        (50.0, 150.0),
        (80.0, 200.0),
        (100.0, 250.0)
    ]
    
    report = {
        "runs": []
    }
    
    for blur in blurs:
        for c1, c2 in canny_pairs:
            for log_hu in [False, True]:
                res = run_experiment_with_config(c1, c2, blur, log_hu)
                
                reals = {k: v for k, v in res.items() if "実写" in k}
                mismatches = [v for k, v in res.items() if "実写" not in k]
                
                mean_mismatch = np.mean(mismatches) if mismatches else 0.0
                max_mismatch = np.max(mismatches) if mismatches else 0.0
                
                run_data = {
                    "config": {
                        "blur": blur,
                        "canny1": c1,
                        "canny2": c2,
                        "log_hu": log_hu
                    },
                    "scores": {
                        "real_near": reals.get("実写直近(ドトール).jpg", 0.0),
                        "real_far": reals.get("実写遠近(ドトール).jpg", 0.0),
                        "mismatches_mean": mean_mismatch,
                        "mismatches_max": max_mismatch
                    }
                }
                report["runs"].append(run_data)
                
    # Sort runs to find best configurations
    # We want high scores for real photos and low scores for mismatches (large gap)
    def score_run(run):
        near = run["scores"]["real_near"]
        far = run["scores"]["real_far"]
        m_max = run["scores"]["mismatches_max"]
        # Formula: average of real photos minus maximum mismatch score
        # But wait! If log_hu is False, the maximum mismatch is bounded at ~0.74, so gap is ~0.
        # If log_hu is True, we want high near and far, e.g. near > 0.8 and far > 0.8
        return (near + far) / 2
        
    sorted_runs = sorted(report["runs"], key=score_run, reverse=True)
    
    print("\n--- Top 5 Configurations by Real Photo Similarity ---")
    for i, run in enumerate(sorted_runs[:5]):
        cfg = run["config"]
        sc = run["scores"]
        print(f"{i+1}. Blur={cfg['blur']}, Canny=({cfg['canny1']}/{cfg['canny2']}), LogHu={cfg['log_hu']}")
        print(f"   Real Near: {sc['real_near']:.4f}, Real Far: {sc['real_far']:.4f}")
        print(f"   Mismatch Mean: {sc['mismatches_mean']:.4f}, Mismatch Max: {sc['mismatches_max']:.4f}")
        
    # Write report
    out_dir = ROOT / "sandbox" / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "param_sweep_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved sweep report to {out_path}")

if __name__ == "__main__":
    main()
