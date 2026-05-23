import cv2
import numpy as np
from typing import Optional
from .config import Config
from .models import MatchResult


class ShapeMatcher:
    def __init__(self, config: Config):
        self.config = config
    
    def match_shapes(self, origin_img: np.ndarray, dummy_img: np.ndarray, method: str = "diff") -> MatchResult:
        """
        形状マッチングによる差分検出を実行する（位置不変）。
        method: "diff" または "matchshapes"
        """
        # 前処理
        origin_processed = self._preprocess(origin_img)
        dummy_processed = self._preprocess(dummy_img)
        
        # 輪郭抽出
        origin_contours = self._extract_contours(origin_processed)
        dummy_contours = self._extract_contours(dummy_processed)
        
        # Huモーメント計算
        hu_origin = self._calculate_hu_moments_from_contours(origin_contours)
        hu_dummy = self._calculate_hu_moments_from_contours(dummy_contours)
        
        # 類似度計算
        if method == "matchshapes":
            similarity = self._compare_contours_match_shapes(origin_contours, dummy_contours)
        else:
            similarity = self._compare_hu_moments(hu_origin, hu_dummy)
        
        # 判定
        is_match = similarity >= self.config.match_threshold
        
        # 新しい要件：視覚的な差分マスクの計算
        h, w = origin_img.shape[:2]
        dummy_resized = dummy_img
        if dummy_img.shape[:2] != (h, w):
            dummy_resized = cv2.resize(dummy_img, (w, h))
            
        diff = cv2.absdiff(origin_img, dummy_resized)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        return MatchResult(
            similarity_score=similarity,
            is_match=is_match,
            origin_path="",
            dummy_path="",
            hu_moments_origin=hu_origin,
            hu_moments_dummy=hu_dummy,
            diff_mask=mask,
            origin_img=origin_img,
            dummy_img=dummy_resized,
            origin_contours=origin_contours,
            dummy_contours=dummy_contours
        )
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        画像の前処理。
        """
        processed = img.copy()
        
        if self.config.grayscale:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        if self.config.gaussian_blur_kernel > 0:
            kernel_size = self.config.gaussian_blur_kernel
            if kernel_size % 2 == 0:
                kernel_size += 1
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)
        
        return processed
    
    def _extract_contours(self, img: np.ndarray) -> list:
        """
        Cannyエッジ検出 + 輪郭抽出。
        """
        # Cannyエッジ検出
        edges = cv2.Canny(img, self.config.canny_threshold1, self.config.canny_threshold2)
        
        # 輪郭抽出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def _contour_hull(self, contours: list) -> Optional[np.ndarray]:
        """複数輪郭を凸包で1つの形状に統合する。"""
        if not contours:
            return None
        all_points = np.vstack([c for c in contours])
        return cv2.convexHull(all_points)

    def _calculate_hu_moments_from_contours(self, contours: list) -> np.ndarray:
        """
        輪郭からHuモーメントを計算。
        複数の輪郭がある場合は、全ての輪郭を統合して1つの形状として扱う。
        """
        hull = self._contour_hull(contours)
        if hull is None:
            return np.zeros(7)

        moments = cv2.moments(hull)
        hu_moments = cv2.HuMoments(moments)
        return hu_moments.flatten()

    def _compare_hu_moments(self, hu1: np.ndarray, hu2: np.ndarray) -> float:
        """Huモーメント差分ベースの類似度（method=diff）。"""
        if hu1 is None or hu2 is None:
            return 0.0

        diff = np.abs(hu1 - hu2)
        max_diff = np.max(diff) if np.max(diff) > 0 else 1.0
        return float(1.0 - (np.mean(diff) / max_diff))

    def _compare_contours_match_shapes(self, contours1: list, contours2: list) -> float:
        """凸包輪郭の cv2.matchShapes による類似度（method=matchshapes）。"""
        hull1 = self._contour_hull(contours1)
        hull2 = self._contour_hull(contours2)
        if hull1 is None or hull2 is None:
            return 0.0

        distance = float(cv2.matchShapes(hull1, hull2, cv2.CONTOURS_MATCH_I1, 0.0))
        return 1.0 / (1.0 + distance)
