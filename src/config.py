from dataclasses import dataclass


@dataclass
class Config:
    # 前処理パラメータ
    grayscale: bool = True
    gaussian_blur_kernel: int = 7
    
    # Cannyエッジ検出パラメータ
    canny_threshold1: float = 30.0
    canny_threshold2: float = 100.0
    
    # 形状マッチング閾値（調整可能）
    match_threshold: float = 0.70  # この値以上ならOKと判定 (IoU基準のデフォルト値)
    
    # 出力設定
    output_dir: str = "output"

    # 実写 → CAD 風正規化（カメラ入力と origin のドメインを揃える）
    photo_normalize_enabled: bool = True
    photo_bottom_crop_ratio: float = 0.26
    photo_size_ratio_threshold: float = 1.5
    photo_bottom_band_removal: bool = True
    photo_remove_band_height_ratio: float = 0.15
    photo_corner_detection_method: str = "rotated"
    photo_alignment_mode: str = "stretch"  # "stretch" または "fit"
    match_method: str = "iou"  # "iou", "diff", "matchshapes"
    hu_moments_compare_limit: int = 7
