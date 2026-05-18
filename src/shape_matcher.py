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
        similarity = self._compare_hu_moments(hu_origin, hu_dummy, method)
        
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
            dummy_img=dummy_resized
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
    
    def _calculate_hu_moments_from_contours(self, contours: list) -> np.ndarray:
        """
        輪郭からHuモーメントを計算。
        複数の輪郭がある場合は、全ての輪郭を統合して1つの形状として扱う。
        """
        if not contours:
            return np.zeros(7)
        
        # 全ての輪郭を統合（凸包を作成）
        all_points = np.vstack([c for c in contours])
        hull = cv2.convexHull(all_points)
        
        # モーメント計算
        moments = cv2.moments(hull)
        
        # Huモーメント計算
        hu_moments = cv2.HuMoments(moments)
        
        return hu_moments.flatten()
    
    def _compare_hu_moments(self, hu1: np.ndarray, hu2: np.ndarray, method: str = "diff") -> float:
        """
        2つのHuモーメントの類似度を計算。
        method: "diff" (差分ベース) または "matchshapes" (OpenCV標準)
        """
        if hu1 is None or hu2 is None:
            return 0.0
        
        if method == "diff":
            # 差分ベース
            diff = np.abs(hu1 - hu2)
            max_diff = np.max(diff) if np.max(diff) > 0 else 1.0
            similarity = 1.0 - (np.mean(diff) / max_diff)
            return float(similarity)
        elif method == "matchshapes":
            # Huモーメントの対数変換（matchShapes用）
            hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
            hu2_log = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
            
            # 差分ベース
            diff = np.abs(hu1_log - hu2_log)
            max_diff = np.max(diff) if np.max(diff) > 0 else 1.0
            similarity = 1.0 - (np.mean(diff) / max_diff)
            return float(similarity)
        else:
            return 0.0
