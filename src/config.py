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
