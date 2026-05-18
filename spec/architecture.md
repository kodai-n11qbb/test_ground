# 画像差分検出プロジェクト - アーキテクチャ設計書

## システムアーキテクチャ

### 全体構成（簡素化版）

```
┌─────────────────────────────────────────────────────────┐
│                    Web Viewer (FastAPI)                 │
├─────────────────────────────────────────────────────────┤
│                    Main Controller                      │
├─────────────────────────────────────────────────────────┤
│  ImageLoader  │  ShapeMatcher  │  ResultExporter        │
├─────────────────────────────────────────────────────────┤
│                    OpenCV Functions                     │
├─────────────────────────────────────────────────────────┤
│                    File I/O                              │
└─────────────────────────────────────────────────────────┘
```

## コンポーネント設計

### 1. ImageLoader
**責務**: 画像ファイルの読み込みとペアリング

```python
class ImageLoader:
    def load_image_pair(self, origin_path: str, dummy_path: str) -> Tuple[np.ndarray, np.ndarray]
    def load_directory(self, origin_dir: str, dummy_dir: str) -> List[Tuple[str, np.ndarray, np.ndarray]]
```

### 2. ShapeMatcher
**責務**: 形状マッチングによる差分検出（位置不変）

```python
class ShapeMatcher:
    def match_shapes(self, origin_img: np.ndarray, dummy_img: np.ndarray, 
                    config: Config) -> MatchResult
    def calculate_hu_moments(self, contour: np.ndarray) -> np.ndarray
    def compare_hu_moments(self, hu1: np.ndarray, hu2: np.ndarray) -> float
```

### 3. ResultExporter
**責務**: 結果の出力（画像ファイル、JSON）

```python
class ResultExporter:
    def export_image(self, result: MatchResult, output_path: str) -> None
    def export_json(self, result: MatchResult, output_path: str) -> None
```

### 4. Web Viewer (api.py)
**責務**: 検出結果の一覧表示と可視化を行うWebインターフェースの提供
- FastAPIによるAPIバックエンドと静的ファイル（出力画像・JSON）の配信
- HTML/JS/CSSによるフロントエンド画面

## データモデル

### MatchResult
```python
@dataclass
class MatchResult:
    similarity_score: float  # 0.0 - 1.0
    is_match: bool  # 閾値による判定
    origin_path: str
    dummy_path: str
    hu_moments_origin: np.ndarray
    hu_moments_dummy: np.ndarray
```

## Dependency Injectionの適用

各コンポーネントは依存をコンストラクタで受け取る：

```python
class DifferenceDetector:
    def __init__(self, contour_comparator: ContourComparator):
        self.contour_comparator = contour_comparator
```

## データフロー

```
1. ImageLoader: 画像読み込み
   ↓
2. ShapeMatcher: 前処理 → 輪郭抽出 → Huモーメント計算 → 形状マッチング
   ↓
3. ResultExporter: 結果出力（画像 + JSON）
```

## 拡張ポイント

### マッチングアルゴリズムの交換可能
- ShapeMatcherに新しい形状マッチング手法を追加可能

### 出力形式の追加
- ResultExporterに新しい出力形式を追加可能

## テスト戦略

### 実験ベースのテスト
- 実際の画像データ（imgs/origin, imgs/dummy）を使用
- 各パラメータ設定での結果を比較
- 目視確認による精度評価

### 簡易的な単体テスト
- 主要関数の基本動作確認
