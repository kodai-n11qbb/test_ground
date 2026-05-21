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

## 未着手
- 透視補正
- 文字セグメントのギャップ結合
- 参照との自動位置合わせ
