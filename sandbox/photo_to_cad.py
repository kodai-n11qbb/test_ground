"""
実写写真を CAD データ風（白背景・紺シルエット）に近づける実験スクリプト。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# 参照 origin の紺に近い BGR
CAD_BLUE_BGR = (80, 40, 20)
CAD_WHITE_BGR = (255, 255, 255)


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def crop_letter_roi(img: np.ndarray, bottom_ratio: float = 0.22) -> np.ndarray:
    """床・足が入る下部を切り落とす。"""
    h = img.shape[0]
    cut = int(h * (1.0 - bottom_ratio))
    return img[:cut, :]


def blue_mask_hsv(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 青ペイント（照明で明るくなる領域も含める）
    lower = np.array([90, 40, 30], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # 低彩度の暗い青
    lower2 = np.array([100, 20, 20], dtype=np.uint8)
    upper2 = np.array([150, 255, 180], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(mask, mask2)


def remove_bottom_band_artifacts(mask: np.ndarray, band_ratio: float = 0.12) -> np.ndarray:
    """マスク下端の細長い連結成分（机縁など）を除去。"""
    h, w = mask.shape[:2]
    y_line = int(h * (1.0 - band_ratio))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if y + bh < y_line:
            continue
        if bw > w * 0.25 and bh < h * 0.04:
            out[labels == i] = 0
    return out


def refine_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    # 小さいゴミ除去
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    min_area = max(80, int(m.size * 0.00002))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def mask_to_flat_cad(mask: np.ndarray, fill_gaps: bool = False) -> np.ndarray:
    """マスクから白背景・紺塗りのフラット画像。"""
    work = mask.copy()
    if fill_gaps:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=2)

    h, w = work.shape[:2]
    flat = np.full((h, w, 3), CAD_WHITE_BGR, dtype=np.uint8)
    flat[work > 0] = CAD_BLUE_BGR
    return flat


def mask_edges_on_white(mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask, 50, 150)
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    out[edges > 0] = CAD_BLUE_BGR
    return out


def auto_crop_to_mask(mask: np.ndarray, pad: int = 20) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = mask.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return x0, y0, x1, y1


def resize_like_reference(flat: np.ndarray, ref_path: Path) -> np.ndarray:
    ref = imread_unicode(ref_path)
    rh, rw = ref.shape[:2]
    return cv2.resize(flat, (rw, rh), interpolation=cv2.INTER_AREA)


def process(
    input_path: Path,
    output_dir: Path,
    ref_path: Path | None,
    bottom_crop: float,
) -> None:
    raw = imread_unicode(input_path)
    cropped = crop_letter_roi(raw, bottom_ratio=bottom_crop)
    mask = remove_bottom_band_artifacts(refine_mask(blue_mask_hsv(cropped)))

    x0, y0, x1, y1 = auto_crop_to_mask(mask)
    mask_c = mask[y0:y1, x0:x1]
    cropped_c = cropped[y0:y1, x0:x1]

    flat = mask_to_flat_cad(mask_c, fill_gaps=False)
    flat_closed = mask_to_flat_cad(mask_c, fill_gaps=True)
    edges = mask_edges_on_white(mask_c)

    stem = input_path.stem
    imwrite_unicode(output_dir / f"{stem}_00_cropped.jpg", cropped)
    imwrite_unicode(output_dir / f"{stem}_01_mask.png", mask_c)
    imwrite_unicode(output_dir / f"{stem}_02_flat.png", flat)
    imwrite_unicode(output_dir / f"{stem}_03_flat_gapfill.png", flat_closed)
    imwrite_unicode(output_dir / f"{stem}_04_edges.png", edges)
    imwrite_unicode(output_dir / f"{stem}_05_overlay.jpg", cv2.addWeighted(cropped_c, 0.35, flat, 0.65, 0))

    if ref_path and ref_path.exists():
        for tag, img in [("flat", flat), ("gapfill", flat_closed)]:
            imwrite_unicode(output_dir / f"{stem}_10_{tag}_vs_ref_size.png", resize_like_reference(img, ref_path))
        ref = imread_unicode(ref_path)
        imwrite_unicode(output_dir / f"{stem}_11_reference_copy.png", ref)

    # パラメータ探索用: 閾値を変えたマスクも1枚
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    _, adapt = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt = refine_mask(adapt)
    imwrite_unicode(output_dir / f"{stem}_90_otsu_inv_debug.png", adapt)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_in = root / "imgs" / "dummy" / "実写直近(ドトール).jpg"
    default_ref = root / "imgs" / "origin" / "POLITEC(ドトール).png"
    default_out = Path(__file__).resolve().parent / "output"

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=default_in)
    p.add_argument("--ref", type=Path, default=default_ref)
    p.add_argument("--output", type=Path, default=default_out)
    p.add_argument("--bottom-crop", type=float, default=0.26)
    args = p.parse_args()

    process(args.input, args.output, args.ref, args.bottom_crop)
    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
