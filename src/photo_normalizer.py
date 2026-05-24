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
        
        # Get origin letter bounding box to align properly
        origin_gray = cv2.cvtColor(origin_bgr, cv2.COLOR_BGR2GRAY)
        _, origin_mask = cv2.threshold(origin_gray, 200, 255, cv2.THRESH_BINARY_INV)
        ys_orig, xs_orig = np.where(origin_mask > 0)
        if len(xs_orig) > 0:
            ox0, ox1 = int(xs_orig.min()), int(xs_orig.max())
            oy0, oy1 = int(ys_orig.min()), int(ys_orig.max())
        else:
            ox0, oy0, ox1, oy1 = 0, 0, ow - 1, oh - 1
        
        if corners is not None:
            # 4隅を整列して射影変換を実行
            src_pts = self._order_points(corners)
            
            if getattr(self.config, "photo_alignment_mode", "stretch") == "stretch":
                dst_pts = np.array([
                    [ox0, oy0],
                    [ox1, oy0],
                    [ox1, oy1],
                    [ox0, oy1]
                ], dtype=np.float32)
            else:
                # Calculate source quad average width and height to check its aspect ratio
                w_top = np.linalg.norm(src_pts[1] - src_pts[0])
                w_bot = np.linalg.norm(src_pts[2] - src_pts[3])
                h_left = np.linalg.norm(src_pts[3] - src_pts[0])
                h_right = np.linalg.norm(src_pts[2] - src_pts[1])
                w_avg = (w_top + w_bot) / 2.0
                h_avg = (h_left + h_right) / 2.0
                src_ar = w_avg / h_avg if h_avg > 0 else 1.0
                
                # Fit vertically or horizontally inside the origin box preserving aspect ratio
                ox_w = ox1 - ox0
                ox_h = oy1 - oy0
                dst_ar = ox_w / ox_h if ox_h > 0 else 1.0
                
                if src_ar > dst_ar:
                    new_h = ox_w / src_ar
                    oy_offset = (ox_h - new_h) / 2.0
                    dst_pts = np.array([
                        [ox0, oy0 + oy_offset],
                        [ox1, oy0 + oy_offset],
                        [ox1, oy0 + oy_offset + new_h],
                        [ox0, oy0 + oy_offset + new_h]
                    ], dtype=np.float32)
                else:
                    new_w = ox_h * src_ar
                    ox_offset = (ox_w - new_w) / 2.0
                    dst_pts = np.array([
                        [ox0 + ox_offset, oy0],
                        [ox0 + ox_offset + new_w, oy0],
                        [ox0 + ox_offset + new_w, oy1],
                        [ox0 + ox_offset, oy1]
                    ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # Use white borderValue to fill margins cleanly
            warped = cv2.warpPerspective(cropped, M, (ow, oh), borderValue=(255, 255, 255))
            
            # Prevent double-processing: use simple _blue_mask_hsv instead of _extract_letter_mask
            warped_mask = self._blue_mask_hsv(warped)
            flat = self._mask_to_flat_cad(warped_mask)
            return flat
        else:
            # フォールバック（従来の外接矩形切り出し＋リサイズ）
            x0, y0, x1, y1 = self._auto_crop_to_mask(mask)
            flat = self._mask_to_flat_cad(mask[y0:y1, x0:x1])
            # Map fallback image directly to the origin bounding box as well for consistency
            resized = cv2.resize(flat, (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_AREA)
            full = np.full((oh, ow, 3), (255, 255, 255), dtype=np.uint8)
            full[oy0:oy1, ox0:ox1] = resized
            return full

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
        method = self.config.photo_corner_detection_method
        if method == "rotated":
            return self._detect_four_corners_rotated(mask)
        elif method == "approx":
            return self._detect_four_corners_approx(mask)
        else:
            return self._detect_four_corners_extremum(mask)

    def _detect_four_corners_extremum(self, mask: np.ndarray) -> np.ndarray | None:
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
        hull = cv2.convexHull(merged_points).reshape(-1, 2)
        
        pts = hull.astype(np.float32)
        s = pts.sum(axis=1)
        diff = pts[:, 0] - pts[:, 1]
        
        tl = hull[np.argmin(s)]
        tr = hull[np.argmax(diff)]
        br = hull[np.argmax(s)]
        bl = hull[np.argmin(diff)]
        
        return np.array([tl, tr, br, bl], dtype=np.int32)

    def _detect_four_corners_approx(self, mask: np.ndarray) -> np.ndarray | None:
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
                
        return self._detect_four_corners_rotated(mask)

    def _detect_four_corners_rotated(self, mask: np.ndarray) -> np.ndarray | None:
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
        
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        return np.array(box, dtype=np.int32)

    def _crop_letter_roi(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        cut = int(h * (1.0 - self.config.photo_bottom_crop_ratio))
        return img[:cut, :]

    def _blue_mask_hsv(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(self.config.hsv_lower1, dtype=np.uint8)
        upper = np.array(self.config.hsv_upper1, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        lower2 = np.array(self.config.hsv_lower2, dtype=np.uint8)
        upper2 = np.array(self.config.hsv_upper2, dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask, mask2)

    def _remove_bottom_band_artifacts(self, mask: np.ndarray) -> np.ndarray:
        if not self.config.photo_bottom_band_removal:
            return mask
        h, w = mask.shape[:2]
        band_ratio = self.config.photo_remove_band_height_ratio
        y_line = int(h * (1.0 - band_ratio))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = mask.copy()
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if y + bh < y_line:
                continue
            if bw > w * 0.25 and bh < h * band_ratio:
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
