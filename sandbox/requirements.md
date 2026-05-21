# sandbox: 実写 → CAD風

## 入力
- `imgs/dummy/実写直近(ドトール).jpg`（POLITEC 実写）
- 参照: `imgs/origin/POLITEC(ドトール).png`（目標の見た目）

## 出力
- `sandbox/output/` に中間画像・最終画像

## やること（段階）
1. 読み込み（日本語パス対応）
2. 下部（床・足）の除外
3. 青文字のマスク（HSV）
4. マスクのノイズ除去・輪郭化
5. 白背景 + CAD色（紺）のフラット合成
6. 参照PNGと同サイズへリサイズ（比較用）

## 成功条件（暫定）
- 木目・筆跡・グレアが消え、2色（白/紺）に近い
- POLITEC の文字塊が読める
- 参照PNGと並べて形状比較できる解像度

## 実行

```powershell
Set-Location sandbox
python photo_to_cad.py
python run_experiments.py
```

## 生成物（`output/`）
| ファイル | 内容 |
|----------|------|
| `*_00_cropped` | 下部カット後 |
| `*_01_mask` | 青抽出マスク |
| `*_02_flat` | 白+紺フラット |
| `*_03_flat_gapfill` | ギャップ閉じ版 |
| `*_04_edges` | 輪郭線のみ |
| `*_10_*_vs_ref_size` | 参照PNG同サイズ |
| `*_edges_match.json` | origin との一致度 |
| `*_12_edges_diff_overlay.png` | 画素差分（赤/緑/黄） |
| `*_origin_edges.png` | origin から抽出した edges |

## edges 一致度検証

```powershell
Set-Location sandbox
python compare_edges.py
```

origin のマスク → Canny と、`*_04_edges` を同サイズ（743×169）で比較。

### 結果（実写直近・2026-05-21 実行）

| 指標 | 値 | 補足 |
|------|-----|------|
| 画素 IoU（シフトなし） | **3.1%** | 1px エッジの重なり |
| 画素 IoU（最良シフト ±24px） | **5.8%** | 最良 dx=7, dy=0 |
| Dice / F1（シフトなし） | **6.0%** | |
| Dice / F1（最良シフト） | **11.0%** | |
| Hu 類似度（diff、edges 同士） | **0.736** | 閾値 0.9 未満 → NG |
| Hu 類似度（matchshapes） | **0.590** | |
| 凸包 matchShapes 距離 | **0.143** | 小さいほど類似（≈0.87 換算） |

画素一致は低い（実写マスクの内外二重線 vs origin の外周一重線、スケール差）。形状ベース（Hu・凸包）は中程度（0.73 前後）。位置合わせ・透視補正なしでは画素では厳しい。

## 未着手
- 透視補正
- 文字セグメントのギャップ結合
- エッジトポロジー揃え（内外線の統一）
