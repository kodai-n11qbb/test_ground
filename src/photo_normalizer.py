"""
カメラ実写を origin（CAD風）と同じ表現域に正規化する。

実写は木目・照明・透視を含み、origin は白背景＋紺シルエットのため、
正規化なしの画素差分・エッジ抽出はドメインギャップで劣化する（前処理の定石）。
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config

CAD_BLUE_BGR = (80, 40, 20)
CAD_WHITE_BGR = (255, 255, 255)


class PhotoNormalizer:
    def __init__(self, config: Config):
        self.config = config

    def needs_normalization(self, dummy_img: np.ndarray, origin_img: np.ndarray) -> bool:
        if not self.config.photo_normalize_enabled:
            return False
        dh, dw = dummy_img.shape[:2]
        oh, ow = origin_img.shape[:2]
        return max(dh, dw) > max(oh, ow) * self.config.photo_size_ratio_threshold

    def normalize(self, photo_bgr: np.ndarray, origin_bgr: np.ndarray) -> np.ndarray:
        """実写を台形補正・CAD 風フラット画像にし、origin と同じ (w, h) にリサイズする。"""
        cropped = self._crop_letter_roi(photo_bgr)
        mask = self._extract_letter_mask(cropped)
        
        corners = self._detect_four_corners(mask)
        oh, ow = origin_bgr.shape[:2]
        
        if corners is not None:
            # 4隅を整列して射影変換を実行
            src_pts = self._order_points(corners)
            dst_pts = np.array([
                [0, 0],
                [ow - 1, 0],
                [ow - 1, oh - 1],
                [0, oh - 1]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(cropped, M, (ow, oh))
            
            # 補正後の画像から再度マスクを抽出してフラット化
            warped_mask = self._extract_letter_mask(warped)
            flat = self._mask_to_flat_cad(warped_mask)
            return flat
        else:
            # フォールバック（従来の外接矩形切り出し＋リサイズ）
            x0, y0, x1, y1 = self._auto_crop_to_mask(mask)
            flat = self._mask_to_flat_cad(mask[y0:y1, x0:x1])
            return cv2.resize(flat, (ow, oh), interpolation=cv2.INTER_AREA)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def _detect_four_corners(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        all_points = []
        for c in contours:
            if cv2.contourArea(c) > 10:
                all_points.append(c)
                
        if not all_points:
            return None
            
        merged_points = np.vstack(all_points)
        hull = cv2.convexHull(merged_points)
        
        peri = cv2.arcLength(hull, True)
        for eps in np.linspace(0.01, 0.2, 50):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
                
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        return np.array(box, dtype=np.int32)

    def _crop_letter_roi(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        cut = int(h * (1.0 - self.config.photo_bottom_crop_ratio))
        return img[:cut, :]

    def _blue_mask_hsv(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([90, 40, 30], dtype=np.uint8)
        upper = np.array([140, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        lower2 = np.array([100, 20, 20], dtype=np.uint8)
        upper2 = np.array([150, 255, 180], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask, mask2)

    def _remove_bottom_band_artifacts(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        y_line = int(h * (1.0 - 0.12))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = mask.copy()
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if y + bh < y_line:
                continue
            if bw > w * 0.25 and bh < h * 0.04:
                out[labels == i] = 0
        return out

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = np.zeros_like(m)
        min_area = max(80, int(m.size * 0.00002))
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def _extract_letter_mask(self, img: np.ndarray) -> np.ndarray:
        return self._remove_bottom_band_artifacts(self._refine_mask(self._blue_mask_hsv(img)))

    def _mask_to_flat_cad(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        flat = np.full((h, w, 3), CAD_WHITE_BGR, dtype=np.uint8)
        flat[mask > 0] = CAD_BLUE_BGR
        return flat

    def _auto_crop_to_mask(self, mask: np.ndarray, pad: int = 20) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            h, w = mask.shape[:2]
            return 0, 0, w, h
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        h, w = mask.shape[:2]
        return (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(w, x1 + pad),
            min(h, y1 + pad),
        )
