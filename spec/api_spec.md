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
実写画像から青色領域の最大輪郭をもとに4隅を特定し、射影変換（台形補正）によって歪みのない長方形にした上で、白背景＋紺シルエットの origin と同じ (幅, 高さ) の BGR 画像を返す。

### ShapeMatcher

#### `__init__(config: Config)`
設定をコンストラクタで受け取る（Dependency Injection）。

#### `match_shapes(origin_img: np.ndarray, dummy_img: np.ndarray, method: str = "diff") -> MatchResult`
形状マッチングによる差分検出を実行する（位置不変）。絶対差分マスクも `MatchResult.diff_mask` に格納する。

**引数**:
- `origin_img`: 元データ画像
- `dummy_img`: 比較対象画像
- `method`: 類似度計算方法
  - `"diff"`: 符号付き対数スケール化を適用したHuモーメント差分ベース
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
結果を画像ファイルとして出力する。元画像とダミー画像のブレンド画像上に、originの輪郭（緑）と補正後dummyの輪郭（赤）を重ねて描画（オーバーレイ）して出力する。

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
    gaussian_blur_kernel: int = 7
    
    # Cannyエッジ検出パラメータ
    canny_threshold1: float = 30.0
    canny_threshold2: float = 100.0
    
    # 形状マッチング閾値（調整可能）
    match_threshold: float = 0.9  # この値以上ならOKと判定
    
    # 出力設定
    output_dir: str = "output"

    # 実写正規化
    photo_normalize_enabled: bool = True
    photo_bottom_crop_ratio: float = 0.26
    photo_size_ratio_threshold: float = 1.5

    # HSV マスク範囲パラメータ（新規追加）
    hsv_lower1: list[int] = (90, 40, 30)
    hsv_upper1: list[int] = (140, 255, 255)
    hsv_lower2: list[int] = (100, 20, 20)
    hsv_upper2: list[int] = (150, 255, 180)

    @classmethod
    def load_from_json(cls, filepath: str = "config.json") -> 'Config':
        """config.jsonから設定をロードする。存在しなければデフォルト値を返す。"""
        pass

    def save_to_json(self, filepath: str = "config.json") -> None:
        """現在のインスタンス状態をconfig.jsonに保存する。"""
        pass
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

### `GET /tuner`
HSVマスクなどのパラメータ調整を行う専用のWebUI画面（HTML）を返します。

### `GET /api/config`
現在の本番設定値（`Config`）のフィールド一覧をJSONで返します。

### `POST /api/tuner/preview`
スライダーから送られてきたHSVパラメータおよび閾値を受け取り、代表テスト画像（ドトール写真）に対する画像処理を一時的に実行します。結果のプレビュー画像を `output/preview/` に保存し、各テスト画像の判定スコアを返します。

**リクエストボディ**:
```json
{
  "hsv_lower1": [90, 40, 30],
  "hsv_upper1": [140, 255, 255],
  "hsv_lower2": [100, 20, 20],
  "hsv_upper2": [150, 255, 180],
  "match_threshold": 0.70
}
```

**戻り値**:
```json
{
  "status": "success",
  "results": {
    "実写直近(ドトール).jpg": { "score": 0.7263, "is_match": true },
    "実写遠近(ドトール).jpg": { "score": 0.7996, "is_match": true },
    "dummy(ドトール)入れ替えのみ_00_TEOPCIL.jpg": { "score": 0.4138, "is_match": false }
  }
}
```

### `POST /api/save-config`
送られてきた設定パラメータを `config.json` に保存し、グローバル設定を更新します。また、本番ディレクトリ内の全画像ペアに対して再判定処理を実行し、結果を `output/` に書き出します。

**リクエストボディ**:
(※ `POST /api/tuner/preview` と同等)

**戻り値**:
```json
{
  "status": "success",
  "message": "Configuration saved and pipeline re-run completed."
}
```

