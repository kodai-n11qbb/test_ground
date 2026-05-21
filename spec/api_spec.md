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

### ShapeMatcher

#### `__init__(config: Config)`
設定をコンストラクタで受け取る（Dependency Injection）。

#### `match_shapes(origin_img: np.ndarray, dummy_img: np.ndarray, method: str = "diff") -> MatchResult`
形状マッチングによる差分検出を実行する（位置不変）。絶対差分マスクも `MatchResult.diff_mask` に格納する。

**引数**:
- `origin_img`: 元データ画像
- `dummy_img`: 比較対象画像
- `method`: 類似度計算方法 (`"diff"` または `"matchshapes"`)

**戻り値**: マッチング結果

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
```

## 使用例

```python
# 設定
config = Config(match_threshold=0.85)

# コンポーネント初期化
loader = ImageLoader()
matcher = ShapeMatcher(config)
exporter = ResultExporter()

# 画像読み込み
origin, dummy = loader.load_image_pair("imgs/origin/test.png", "imgs/dummy/test.png")

# 形状マッチング
result = matcher.match_shapes(origin, dummy, method="diff")

# 結果出力
exporter.export_image(result, "output/result.png")
exporter.export_json(result, "output/result.json")

# 結果確認
print(f"Similarity: {result.similarity_score:.3f}")
print(f"Match: {result.is_match}")
```

## Web API仕様 (FastAPI)

`api.py` にて提供されるWebビューア用のAPIエンドポイントです。

### `GET /`
差分結果を閲覧できるWebインターフェース（HTML）を返します。

### `GET /api/results`
`output/` ディレクトリ内に保存されているすべての差分検出結果（JSON）と、対応する画像パスのリストを返します。（類似度の高い順にソート済）

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
      "hu_moments_origin": [...],
      "hu_moments_dummy": [...],
      "result_image": "/output/dummy(ドトール)入れ替えズレあり_00_CLPOIET_result.png"
    }
  ]
}
```
