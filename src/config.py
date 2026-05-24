from dataclasses import dataclass, asdict, field
import json
import os


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
    
    # 入力設定
    origin_dir: str = "imgs/origin"
    dummy_dir: str = "imgs/dummy"

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

    # HSV マスクパラメータ
    hsv_lower1: list[int] = field(default_factory=lambda: [90, 40, 30])
    hsv_upper1: list[int] = field(default_factory=lambda: [140, 255, 255])
    hsv_lower2: list[int] = field(default_factory=lambda: [100, 20, 20])
    hsv_upper2: list[int] = field(default_factory=lambda: [150, 255, 180])

    @classmethod
    def load_from_json(cls, filepath: str = "config.json") -> "Config":
        """config.jsonが存在する場合は読み込み、無ければデフォルト値を生成する。"""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                import inspect
                sig = inspect.signature(cls)
                valid_keys = {k for k in sig.parameters.keys()}
                filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                return cls(**filtered_data)
            except Exception as e:
                print(f"Warning: Failed to load config.json ({e}). Using defaults.")
        return cls()

    def save_to_json(self, filepath: str = "config.json") -> None:
        """現在の設定値をJSONファイルに書き出す。"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config.json: {e}")
