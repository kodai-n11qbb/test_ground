# 画像差分検出プロジェクト - 実装計画

## 実装フェーズ

### フェーズ1: プロジェクトセットアップ
- [ ] Python仮想環境の作成
- [ ] 依存パッケージのインストール（opencv-python, numpy）
- [ ] プロジェクトディレクトリ構造の作成
- [ ] requirements.txtの作成

### フェーズ2: 基本コンポーネント実装
- [ ] Configクラスの実装
- [ ] MatchResultクラスの実装
- [ ] ImageLoaderの実装
- [ ] ShapeMatcherの実装（Huモーメント、形状マッチング）
- [ ] ResultExporterの実装

### フェーズ3: メイン処理実装
- [ ] main.pyの実装
- [ ] コマンドライン引数の処理

### フェーズ4: 実験と調整
- `[x]` 実画像データを使った実験
- `[x]` パラメータ調整（match_threshold, canny_threshold）
- `[x]` 2つの類似度計算方法の比較
- `[x]` 結果の目視確認
- `[x]` 最適な方法の採用（method="diff", threshold=0.73を採用）

### フェーズ5: ドキュメント更新・Web UI実装
- `[x]` README.mdの作成
- `[x]` 使用方法の記載
- `[x]` 絶対差分による赤色ハイライト描画の実装
- `[x]` FastAPIとHTML/JSによるWebビューアの実装

## ディレクトリ構造（簡素化）

```
test_ground/
├── imgs/
│   ├── origin/
│   └── dummy/
├── spec/
│   ├── requirements.md
│   ├── architecture.md
│   ├── api_spec.md
│   ├── test_spec.md
│   └── implementation_plan.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── image_loader.py
│   ├── shape_matcher.py
│   └── result_exporter.py
├── output/
├── main.py
├── requirements.txt
├── DEV_POLICY.mmd
└── README.md
```

## 依存パッケージ

```
opencv-python>=4.8.0
numpy>=1.24.0
```

## 実装の優先順位

1. **基本コンポーネント**: Config, Models
2. **画像処理**: ImageLoader, ShapeMatcher
3. **出力**: ResultExporter
4. **メイン処理**: main.py
5. **実験**: 実画像データでパラメータ調整

## DEV_POLICY.mmdの適用

### Dependency Injection
- ShapeMatcherにConfigを注入
- 必要に応じて依存をコンストラクタで注入

### Rule of Three
- 同じコードが3回出るまで共通化しない
- YAGNIの原則を徹底

### 実験ベースのアプローチ
- 実画像データで結果を確認
- パラメータ調整を繰り返す

## 次のステップ

1. この実装計画を確認
2. フェーズ1から開始
3. 実画像データ（imgs/origin, imgs/dummy）を使用して実験
