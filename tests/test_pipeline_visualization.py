import cv2
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.image_loader import ImageLoader
from src.shape_matcher import ShapeMatcher
from src.models import MatchResult


class PipelineVisualizer:
    def __init__(self, matcher: ShapeMatcher, output_dir: str):
        self.matcher = matcher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_pipeline(self, origin_img: np.ndarray, dummy_img: np.ndarray, 
                          result_name: str) -> None:
        origin_processed = self.matcher._preprocess(origin_img)
        dummy_processed = self.matcher._preprocess(dummy_img)
        
        origin_edges = self.matcher._extract_contours(origin_processed)
        dummy_edges = self.matcher._extract_contours(dummy_processed)
        
        origin_hull = self._create_convex_hull_from_contours(origin_edges)
        dummy_hull = self._create_convex_hull_from_contours(dummy_edges)
        
        h, w = origin_img.shape[:2]
        dummy_resized = dummy_img
        if dummy_img.shape[:2] != (h, w):
            dummy_resized = cv2.resize(dummy_img, (w, h))
        
        diff_mask = self._calculate_diff_mask(origin_img, dummy_resized)
        
        self._save_edges(origin_processed, self.output_dir / f"{result_name}_origin_edges.png")
        self._save_edges(dummy_processed, self.output_dir / f"{result_name}_dummy_edges.png")
        self._save_contour_hull(origin_hull, (h, w), self.output_dir / f"{result_name}_origin_hull.png")
        self._save_contour_hull(dummy_hull, (h, w), self.output_dir / f"{result_name}_dummy_hull.png")
        self._save_image(diff_mask, self.output_dir / f"{result_name}_diff_mask.png")
    
    def _create_convex_hull_from_contours(self, contours: list) -> np.ndarray:
        if not contours:
            return np.array([])
        
        all_points = np.vstack([c for c in contours])
        hull = cv2.convexHull(all_points)
        return hull
    
    def _calculate_diff_mask(self, origin_img: np.ndarray, dummy_img: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(origin_img, dummy_img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        return mask
    
    def _save_edges(self, processed_img: np.ndarray, path: Path) -> None:
        edges = cv2.Canny(processed_img, self.matcher.config.canny_threshold1, 
                         self.matcher.config.canny_threshold2)
        cv2.imwrite(str(path), edges)
    
    def _save_image(self, img: np.ndarray, path: Path) -> None:
        cv2.imwrite(str(path), img)
    
    def _save_contour_hull(self, hull: np.ndarray, shape: tuple, path: Path) -> None:
        if hull.size == 0:
            blank = np.zeros(shape, dtype=np.uint8)
            cv2.imwrite(str(path), blank)
            return
        
        blank = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(blank, [hull], -1, 255, 2)
        cv2.imwrite(str(path), blank)


def test_pipeline_visualization():
    config = Config(
        match_threshold=0.73,
        canny_threshold1=50.0,
        canny_threshold2=150.0
    )
    
    loader = ImageLoader()
    matcher = ShapeMatcher(config)
    visualizer = PipelineVisualizer(matcher, "tests/output")
    
    origin_dir = "imgs/origin"
    dummy_dir = "imgs/dummy"
    
    pairs = loader.load_directory(origin_dir, dummy_dir)
    
    for origin_file, dummy_file, origin_img, dummy_img in pairs:
        result_name = dummy_file.replace('.', '_')
        print(f"Processing: {dummy_file}")
        
        visualizer.visualize_pipeline(origin_img, dummy_img, result_name)
        
        print(f"  Saved visualization to tests/output/{result_name}_*.png")


if __name__ == "__main__":
    test_pipeline_visualization()
