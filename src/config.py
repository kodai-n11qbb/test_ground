from dataclasses import dataclass


@dataclass
class Config:
    # 前処理パラメータ
    grayscale: bool = True
    gaussian_blur_kernel: int = 5
    
    # Cannyエッジ検出パラメータ
    canny_threshold1: float = 50.0
    canny_threshold2: float = 150.0
    
    # 形状マッチング閾値（調整可能）
    match_threshold: float = 0.9  # この値以上ならOKと判定
    
    # 出力設定
    output_dir: str = "output"

    # 実写 → CAD 風正規化（カメラ入力と origin のドメインを揃える）
    photo_normalize_enabled: bool = True
    photo_bottom_crop_ratio: float = 0.26
    photo_size_ratio_threshold: float = 1.5
