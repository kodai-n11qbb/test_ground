# 画像差分検出プロジェクト - アーキテクチャ設計書

## システムアーキテクチャ

### 全体構成（簡素化版）

```
┌─────────────────────────────────────────────────────────┐
│                    Web Viewer (FastAPI)                 │
├─────────────────────────────────────────────────────────┤
│              Main Controller (MatchPipeline)              │
├─────────────────────────────────────────────────────────┤
│  ImageLoader  │  PhotoNormalizer │  ShapeMatcher │ ResultExporter │
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

### 2. PhotoNormalizer
**責務**: カメラ実写 dummy の看板領域から4隅を特定し、射影変換（台形補正）によって origin と同じ CAD 風表現（サイズ・縦横比も一致）に正規化する

```python
class PhotoNormalizer:
    def __init__(self, config: Config)
    def needs_normalization(self, dummy_img: np.ndarray, origin_img: np.ndarray) -> bool
    def normalize(self, photo_bgr: np.ndarray, origin_bgr: np.ndarray) -> np.ndarray
```

- `Config` を DI で受け取る
- 解像度比が閾値以下の dummy（合成データ等）はスキップ
- 青色マスクの輪郭検出から看板の4隅を自動抽出し（台形極値法/approxPolyDP/RotatedRectのハイブリッド）、`warpPerspective` を用いて、originの文字バウンディングボックスの中にアスペクト比を保存してフィットさせる
- 射影変換後のマスク抽出において、余分なモルフォロジー演算の重複を排除する

### 3. MatchPipeline
**責務**: 読み込み・正規化・照合・出力のオーケストレーション（DIで各コンポーネントを受け取る）

```python
class MatchPipeline:
    def __init__(self, loader, normalizer, matcher, exporter)
    def prepare_dummy_for_match(self, origin_img, dummy_img) -> Tuple[np.ndarray, bool]
    def process_pair(self, origin_img, dummy_img, method="diff") -> MatchResult
```

`pipeline_factory.build_default_pipeline(config)` が composition root 用の既定配線を提供する。

### 4. ShapeMatcher
**責務**: 形状マッチングによる差分検出（位置不変、符号付き対数スケールHuモーメント比較によるスケール差補正）およびアライメント調整
* 主要な最初の2つのHuモーメント（`hu[:2]`）のみを比較して、レンズや解像度のギザギザによる微細な高次ノイズを排除した頑健なマッチングを行う。

```python
class ShapeMatcher:
    def __init__(self, config: Config)
    def match_shapes(self, origin_img: np.ndarray, dummy_img: np.ndarray,
                    method: str = "diff") -> MatchResult
```

### 5. ResultExporter
**責務**: 結果の出力（画像ファイル、JSON）
- 出力画像には、originの輪郭（例：緑）と補正後dummyの輪郭（例：赤）を重ねて描画（オーバーレイ）し、ズレの状態が視覚的にわかるようにする

```python
class ResultExporter:
    def export_image(self, result: MatchResult, output_path: str) -> None
    def export_json(self, result: MatchResult, output_path: str) -> None
```

### 6. Web Viewer (api.py)
**責務**: 検出結果の一覧表示と可視化を行うWebインターフェースの提供、およびパラメータ調整用インターフェース（Tuner）の提供
- FastAPIによるAPIバックエンドと静的ファイル（出力画像・JSON）の配信
- HTML/JS/CSSによるフロントエンド画面（結果閲覧画面・チューニング画面）
- パラメータ調整時のプレビュー生成および設定保存（`config.json` への永続化）
- 保存時にすべての本番画像ペアに対する照合パイプラインの再実行管理

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
    photo_normalized: bool
```

## Dependency Injectionの適用

各コンポーネントは依存をコンストラクタで受け取る：

```python
pipeline = build_default_pipeline(Config(match_threshold=0.73))
```

## データフロー

### 1. 本番一括バッチ処理フロー
```
1. MatchPipeline / ImageLoader: 画像読み込み
   ↓
2. MatchPipeline / PhotoNormalizer: 実写なら4隅検出・射影変換（台形補正）による長方形化およびCAD風正規化
   ↓
3. MatchPipeline / ShapeMatcher: 前処理 → 輪郭抽出および重ね合わせアライメント調整 → 類似度算出
   ↓
4. MatchPipeline / ResultExporter: 結果出力（輪郭同士を重ね合わせた画像 + JSON）
```

### 2. スライダー調整時のプレビューデータフロー
```
1. tuner.html: ユーザーがスライダーを操作（HSV/閾値）
   ↓ (250ms デバウンス)
2. tuner.html / api.py: POST /api/tuner/preview へパラメータを送信
   ↓
3. api.py / MatchPipeline: 代表テスト画像（ドトール写真）に対して、一時的なパラメータでメモリ上パイプライン処理を実行
   ↓
4. ResultExporter: プレビュー画像を output/preview/ ディレクトリへ出力・上書き
   ↓
5. api.py -> tuner.html: 各テスト画像のIoUスコアをJSONで返却
   ↓
6. tuner.html: 画像要素のsrc末尾にキャッシュバスター（?t=timestamp）を付与してリロードし、表示を瞬時に更新
```

### 3. 設定保存および本番パイプライン再実行のデータフロー
```
1. tuner.html: 「設定を保存（適用）」ボタンを押下
   ↓
2. tuner.html / api.py: POST /api/save-config を呼び出し
   ↓
3. api.py / Config: 送信されたパラメータを config.json に永続化保存し、アプリ内のグローバル Config を更新
   ↓
4. api.py / MatchPipeline: 本番用の全画像ペア（imgs/）に対して順次照合処理を実行
   ↓
5. ResultExporter: 結果ファイル（output/*_result.png / json）を新しい設定で上書き保存
   ↓
6. api.py -> tuner.html: 処理完了レスポンスを返却
   ↓
7. tuner.html: メインビューア画面（/）へリダイレクトし、新しい設定値での結果を表示
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
