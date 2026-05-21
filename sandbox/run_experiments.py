"""複数パラメータで一括実行。"""
from pathlib import Path

from photo_to_cad import process

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "imgs" / "dummy" / "実写直近(ドトール).jpg"
REF = ROOT / "imgs" / "origin" / "POLITEC(ドトール).png"
OUT = Path(__file__).resolve().parent / "output"

if __name__ == "__main__":
    for ratio, name in [(0.18, "crop18"), (0.22, "crop22"), (0.26, "crop26")]:
        process(INPUT, OUT / name, REF, bottom_crop=ratio)
    print("batch done")
