# 画像差分検出プロジェクト - API仕様書

## パブリックAPI

### ImageLoader

#### `load_image_pair(origin_path: str, dummy_path: str) -> Tuple[np.ndarray, np.ndarray]`
指定されたパスの画像ペアを読み込む。

**引数**:
- `origin_path`: 元データのファイルパス
- `dummy_path`: 比較対象のファイルパス

**戻り値**: (origin_image, dummy_image) のタプル（OpenCV画像配列）

#### `load_directory(origin_dir: str, dummy_dir: str) -> List[Tuple[str, str, np.ndarray, np.ndarray]]`
ディレクトリ内の全画像を読み込み、ファイル名に含まれる「カッコ内の文字列（キーワード）」が一致するすべての組み合わせをペアリングして読み込む。

**引数**:
- `origin_dir`: 元データディレクトリパス
- `dummy_dir`: 比較対象ディレクトリパス

**戻り値**: (origin_file, dummy_file, origin_image, dummy_image) のリスト

### PhotoNormalizer

#### `__init__(config: Config)`
設定をコンストラクタで受け取る（Dependency Injection）。

#### `needs_normalization(dummy_img: np.ndarray, origin_img: np.ndarray) -> bool`
実写の CAD 風正規化が必要か判定する。解像度比が `photo_size_ratio_threshold` 以下なら `False`。

#### `normalize(photo_bgr: np.ndarray, origin_bgr: np.ndarray) -> np.ndarray`
実写を白背景＋紺シルエットにし、origin と同じ (幅, 高さ) の BGR 画像を返す。

### ShapeMatcher

#### `__init__(config: Config)`
設定をコンストラクタで受け取る（Dependency Injection）。

#### `match_shapes(origin_img: np.ndarray, dummy_img: np.ndarray, method: str = "diff") -> MatchResult`
形状マッチングによる差分検出を実行する（位置不変）。絶対差分マスクも `MatchResult.diff_mask` に格納する。

**引数**:
- `origin_img`: 元データ画像
- `dummy_img`: 比較対象画像
- `method`: 類似度計算方法
  - `"diff"`: Huモーメント差分ベース
  - `"matchshapes"`: 凸包輪郭の `cv2.matchShapes`（I1距離 → `1/(1+d)`）

**戻り値**: マッチング結果

### MatchPipeline

#### `__init__(loader, normalizer, matcher, exporter)`
各コンポーネントをコンストラクタで受け取る（Dependency Injection）。

#### `prepare_dummy_for_match(origin_img, dummy_img) -> Tuple[np.ndarray, bool]`
必要なら CAD 風正規化した dummy と `photo_normalized` フラグを返す。

#### `process_pair(origin_img, dummy_img, method="diff") -> MatchResult`
正規化 → 形状マッチングまで実行する。

#### `create_match_pipeline(loader, normalizer, matcher, exporter) -> MatchPipeline`
依存を外から渡してパイプラインを組み立てる。

### ResultExporter

#### `export_image(result: MatchResult, output_path: str) -> None`
結果を画像ファイルとして出力する。

**引数**:
- `result`: マッチング結果
- `output_path`: 出力ファイルパス

#### `export_json(result: MatchResult, output_path: str) -> None`
結果をJSONファイルとして出力する。

**引数**:
- `result`: マッチング結果
- `output_path`: 出力ファイルパス

## コンフィギュレーション

### Config

```python
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

    # 実写正規化
    photo_normalize_enabled: bool = True
    photo_bottom_crop_ratio: float = 0.26
    photo_size_ratio_threshold: float = 1.5
```

## 使用例

```python
from src.config import Config
from src.image_loader import ImageLoader
from src.photo_normalizer import PhotoNormalizer
from src.shape_matcher import ShapeMatcher
from src.result_exporter import ResultExporter
from src.match_pipeline import create_match_pipeline

config = Config(match_threshold=0.85)
pipeline = create_match_pipeline(
    loader=ImageLoader(),
    normalizer=PhotoNormalizer(config),
    matcher=ShapeMatcher(config),
    exporter=ResultExporter(),
)

origin, dummy = pipeline.loader.load_image_pair(
    "imgs/origin/test.png", "imgs/dummy/test.png"
)
result = pipeline.process_pair(origin, dummy, method="diff")
result.origin_path = "imgs/origin/test.png"
result.dummy_path = "imgs/dummy/test.png"

pipeline.export_result(result, "output/result.png", "output/result.json")
print(f"Similarity: {result.similarity_score:.3f}")
print(f"Photo normalized: {result.photo_normalized}")
print(f"Match: {result.is_match}")
```

## Web API仕様 (FastAPI)

`api.py` にて提供されるWebビューア用のAPIエンドポイントです。

### `GET /`
差分結果を閲覧できるWebインターフェース（HTML）を返します。

### `GET /api/result-image/{result_id}`
結果 PNG を返す。`result_id` は JSON のベース名（例: `実写直近(ドトール)_jpg`、URL エンコード済み）。日本語ファイル名は `/output/` 直リンクではなく本 API 経由で配信する。

### `GET /api/results`
`output/` ディレクトリ内に保存されているすべての差分検出結果（JSON）と、対応する画像パスのリストを返します。（類似度の高い順にソート済）

`result_image` は `/api/result-image/{urlencoded_id}` 形式。

**戻り値**:
```json
{
  "results": [
    {
      "id": "dummy(ドトール)入れ替えズレあり_00_CLPOIET",
      "similarity_score": 0.739,
      "is_match": true,
      "origin_path": "imgs/origin/POLITEC(ドトール).png",
      "dummy_path": "imgs/dummy/dummy(ドトール)入れ替えズレあり_00_CLPOIET.jpg",
      "photo_normalized": false,
      "hu_moments_origin": [...],
      "hu_moments_dummy": [...],
      "result_image": "/api/result-image/dummy(%E3%83%89%E3%83%88%E3%83%BC%E3%83%AB)%E5%85%A5%E3%82%8C%E6%9B%BF%E3%81%88%E3%82%9A%E3%83%AC%E3%81%82%E3%82%8A_00_CLPOIET_jpg"
    }
  ]
}
```
