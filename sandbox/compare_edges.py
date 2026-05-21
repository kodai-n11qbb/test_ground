"""
実写由来 edges（*_04_edges.png）と origin の edges を形状・画素で比較する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.shape_matcher import ShapeMatcher  # noqa: E402

from photo_to_cad import imread_unicode, imwrite_unicode  # noqa: E402

CANNY_LO, CANNY_HI = 50, 150
SHIFT_RANGE = 24


def origin_blue_mask(bgr: np.ndarray) -> np.ndarray:
    """origin PNG（紺塗り）から文字マスク。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)


def edges_from_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.Canny(mask, CANNY_LO, CANNY_HI)


def edges_from_edges_png(bgr: np.ndarray) -> np.ndarray:
    """*_04_edges.png（白地・紺線）から 1px エッジマップ。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, edge = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    return edge


def edges_to_bgr(edge: np.ndarray) -> np.ndarray:
    out = np.full((*edge.shape, 3), 255, dtype=np.uint8)
    out[edge > 0] = (80, 40, 20)
    return out


def pixel_overlap(a: np.ndarray, b: np.ndarray) -> dict:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    sum_a, sum_b = int(a.sum()), int(b.sum())
    iou = inter / union if union else 0.0
    dice = (2 * inter) / (sum_a + sum_b) if (sum_a + sum_b) else 0.0
    prec = inter / sum_b if sum_b else 0.0
    rec = inter / sum_a if sum_a else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "f1": round(f1, 4),
        "precision_photo_vs_origin": round(prec, 4),
        "recall_photo_covers_origin": round(rec, 4),
        "origin_edge_pixels": sum_a,
        "photo_edge_pixels": sum_b,
        "overlap_pixels": inter,
    }


def best_shift_overlap(origin_e: np.ndarray, photo_e: np.ndarray, radius: int) -> dict:
    best = pixel_overlap(origin_e, photo_e)
    best["shift_dx"] = 0
    best["shift_dy"] = 0
    h, w = origin_e.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(photo_e, M, (w, h), borderValue=0)
            m = pixel_overlap(origin_e, shifted)
            if m["iou"] > best["iou"]:
                best = {**m, "shift_dx": dx, "shift_dy": dy}
    return best


def hu_similarity(origin_bgr: np.ndarray, photo_bgr: np.ndarray) -> dict:
    sm = ShapeMatcher(Config())
    out = {}
    for method in ("diff", "matchshapes"):
        r = sm.match_shapes(origin_bgr, photo_bgr, method=method)
        out[method] = {
            "similarity": round(float(r.similarity_score), 4),
            "is_match_threshold_0.9": bool(r.is_match),
        }
    return out


def contour_match_shapes(origin_e: np.ndarray, photo_e: np.ndarray) -> dict:
    """全エッジ点の凸包輪郭で cv2.matchShapes（値が小さいほど似ている）。"""
    def hull_contour(edge: np.ndarray):
        ys, xs = np.where(edge > 0)
        if len(xs) == 0:
            return None
        pts = np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        return hull

    h1, h2 = hull_contour(origin_e), hull_contour(photo_e)
    if h1 is None or h2 is None:
        return {"match_shapes_distance": None, "similarity_from_distance": 0.0}

    dist = float(cv2.matchShapes(h1, h2, cv2.CONTOURS_MATCH_I1, 0.0))
    # 経験的に 0=同一、~0.01 近い、>0.1 別形状。レポート用に 0-1 へ変換
    sim = 1.0 / (1.0 + dist)
    return {
        "match_shapes_distance": round(dist, 6),
        "similarity_from_distance": round(sim, 4),
    }


def diff_overlay(origin_e: np.ndarray, photo_e: np.ndarray) -> np.ndarray:
    o = origin_e > 0
    p = photo_e > 0
    vis = np.zeros((*origin_e.shape, 3), dtype=np.uint8)
    vis[o & ~p] = (0, 0, 255)      # origin のみ: 赤
    vis[~o & p] = (0, 200, 0)      # photo のみ: 緑
    vis[o & p] = (0, 220, 220)     # 一致: 黄
    return vis


def compare(
    photo_edges_path: Path,
    origin_path: Path,
    output_dir: Path,
    stem: str,
) -> dict:
    origin = imread_unicode(origin_path)
    photo_edges_img = imread_unicode(photo_edges_path)

    origin_mask = origin_blue_mask(origin)
    origin_e_full = edges_from_mask(origin_mask)
    photo_e_full = edges_from_edges_png(photo_edges_img)

    rh, rw = origin.shape[:2]
    photo_e = cv2.resize(photo_e_full, (rw, rh), interpolation=cv2.INTER_NEAREST)
    origin_e = origin_e_full  # 既に origin サイズ

    imwrite_unicode(output_dir / f"{stem}_origin_edges.png", edges_to_bgr(origin_e))
    imwrite_unicode(output_dir / f"{stem}_photo_edges_refsize.png", edges_to_bgr(photo_e))
    imwrite_unicode(output_dir / f"{stem}_12_edges_diff_overlay.png", diff_overlay(origin_e, photo_e))

    shifted_photo = photo_e.copy()
    best = best_shift_overlap(origin_e, photo_e, SHIFT_RANGE)
    if best["shift_dx"] or best["shift_dy"]:
        M = np.float32([[1, 0, best["shift_dx"]], [0, 1, best["shift_dy"]]])
        shifted_photo = cv2.warpAffine(photo_e, M, (rw, rh), borderValue=0)
        imwrite_unicode(
            output_dir / f"{stem}_13_edges_diff_overlay_aligned.png",
            diff_overlay(origin_e, shifted_photo),
        )

    report = {
        "origin": str(origin_path.relative_to(ROOT)),
        "photo_edges": str(photo_edges_path.relative_to(ROOT)),
        "ref_size_wh": [int(rw), int(rh)],
        "pixel_no_shift": pixel_overlap(origin_e, photo_e),
        "pixel_best_shift": best,
        "hu_moments_on_edge_images_refsize": hu_similarity(
            edges_to_bgr(origin_e), edges_to_bgr(photo_e)
        ),
        "hu_moments_on_origin_png_vs_photo_edges_refsize": hu_similarity(
            origin, edges_to_bgr(photo_e)
        ),
        "contour_match_shapes_hull": contour_match_shapes(origin_e, photo_e),
        "contour_match_shapes_hull_aligned": contour_match_shapes(origin_e, shifted_photo),
        "legend": {
            "red": "origin only",
            "green": "photo only",
            "yellow": "overlap",
        },
        "notes": [
            "pixel_* は 1px エッジの重なり（位置ずれに弱い）",
            "hu_* は凸包統合 Hu（位置・平行移動に比較的強い）",
            "match_shapes_distance は凸包輪郭の I1 距離（小さいほど類似）",
        ],
    }

    out_json = output_dir / f"{stem}_edges_match.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    stem = "実写直近(ドトール)"
    origin = ROOT / "imgs" / "origin" / "POLITEC(ドトール).png"
    photo_edges = Path(__file__).resolve().parent / "output" / f"{stem}_04_edges.png"
    output_dir = Path(__file__).resolve().parent / "output"

    report = compare(photo_edges, origin, output_dir, stem)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
