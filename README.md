# 画像差分検出ツール

本プロジェクトは、元データ（CADデータ想定）とテストデータ（カメラ入力想定）の画像間で、位置ズレを許容しつつ形状の差分を検出するシステムです。
また、出力された結果をWebブラウザ上で確認できるFastAPIベースのWebビューアを備えています。

## 技術スタック
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- pytest (`pytest`)

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

基本的な実行方法：
```bash
python main.py
```

各種パラメータの指定：
```bash
python main.py --origin imgs/origin --dummy imgs/dummy --threshold 0.73 --method diff
```

### オプション
- `--origin`: 元データのディレクトリ（デフォルト: `imgs/origin`）
- `--dummy`: 比較対象データのディレクトリ（デフォルト: `imgs/dummy`）
- `--threshold`: マッチングの閾値。この値以上であれば「Match」と判定（デフォルト: `0.73`）
- `--method`: 類似度の計算方法。`diff` または `matchshapes`（デフォルト: `diff`）
- `--output`: 検出結果画像の出力ディレクトリ（デフォルト: `output`）

### 画像のペアリング仕様
ファイル名に含まれる「カッコ内の文字列（全角・半角問わず）」をキーワードとして抽出し、同じキーワードを持つ元画像とテスト画像を自動的にペアリングします。1つの元画像に対して複数のテスト画像が存在する「1対多」の構成にも対応しています。

## Webビューアの起動

出力結果をブラウザで視覚的に確認するためのビューアを提供しています。

```bash
python api.py
```

起動後、ブラウザで `http://127.0.0.1:8000/` にアクセスしてください。
MATCH/NO MATCHのフィルタリングや、赤くハイライトされた差分画像の一覧表示が可能です。

## テストの実行

`DEV_POLICY.md`の「Refactor-ready Test」方針に基づき、各コンポーネントのテストを実装しています。

```bash
pytest tests/
```

## 出力内容
`output/` フォルダ内に、比較ペアごとに以下のファイルが出力されます：
- `*_result.png` : 判定結果と類似度を描画した画像ファイル
- `*_result.json` : 各種スコア・結果を含むJSONファイル
