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
    def load_directory(self, origin_dir: str, dummy_dir: str) -> List[Tuple[str, str, np.ndarray, np.ndarray]]
```

### 2. ShapeMatcher
**責務**: 形状マッチングによる差分検出（位置不変）

```python
class ShapeMatcher:
    def __init__(self, config: Config)
    def match_shapes(self, origin_img: np.ndarray, dummy_img: np.ndarray,
                    method: str = "diff") -> MatchResult
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
    hu_moments_origin: Optional[np.ndarray]
    hu_moments_dummy: Optional[np.ndarray]
    diff_mask: Optional[np.ndarray]
    origin_img: Optional[np.ndarray]
    dummy_img: Optional[np.ndarray]
```

## Dependency Injectionの適用

各コンポーネントは依存をコンストラクタで受け取る：

```python
config = Config(match_threshold=0.73)
matcher = ShapeMatcher(config)
exporter = ResultExporter()
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

- `tests/` に pytest による単体テスト（合成画像）
- 実画像での検証は `python main.py` 実行後、`output/` と Web ビューアで確認
- [DEV_POLICY.md](../DEV_POLICY.md) の Refactor-ready Test に従う
