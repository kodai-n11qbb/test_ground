import cv2
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.image_loader import ImageLoader
from src.match_pipeline import create_match_pipeline
from src.photo_normalizer import PhotoNormalizer
from src.result_exporter import ResultExporter
from src.shape_matcher import ShapeMatcher


class PipelineVisualizer:
    def __init__(self, matcher: ShapeMatcher, normalizer: PhotoNormalizer, output_dir: str):
        self.matcher = matcher
        self.normalizer = normalizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_pipeline(
        self, origin_img: np.ndarray, dummy_img: np.ndarray, result_name: str
    ) -> None:
        dummy_for_match, _ = (
            (self.normalizer.normalize(dummy_img, origin_img), True)
            if self.normalizer.needs_normalization(dummy_img, origin_img)
            else (dummy_img, False)
        )

        origin_processed = self.matcher._preprocess(origin_img)
        dummy_processed = self.matcher._preprocess(dummy_for_match)

        origin_edges = self.matcher._extract_contours(origin_processed)
        dummy_edges = self.matcher._extract_contours(dummy_processed)

        origin_hull = self.matcher._contour_hull(origin_edges)
        dummy_hull = self.matcher._contour_hull(dummy_edges)

        h, w = origin_img.shape[:2]
        dummy_resized = dummy_for_match
        if dummy_for_match.shape[:2] != (h, w):
            dummy_resized = cv2.resize(dummy_for_match, (w, h))

        diff_mask = self._calculate_diff_mask(origin_img, dummy_resized)

        self._save_edges(origin_processed, self.output_dir / f"{result_name}_origin_edges.png")
        self._save_edges(dummy_processed, self.output_dir / f"{result_name}_dummy_edges.png")
        self._save_contour_hull(origin_hull, (h, w), self.output_dir / f"{result_name}_origin_hull.png")
        self._save_contour_hull(dummy_hull, (h, w), self.output_dir / f"{result_name}_dummy_hull.png")
        self._save_image(diff_mask, self.output_dir / f"{result_name}_diff_mask.png")

    def _calculate_diff_mask(self, origin_img: np.ndarray, dummy_img: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(origin_img, dummy_img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        return mask

    def _save_array(self, img: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(path.suffix or ".png", img)
        assert ok
        buf.tofile(str(path))

    def _save_edges(self, processed_img: np.ndarray, path: Path) -> None:
        edges = cv2.Canny(
            processed_img,
            self.matcher.config.canny_threshold1,
            self.matcher.config.canny_threshold2,
        )
        self._save_array(edges, path)

    def _save_image(self, img: np.ndarray, path: Path) -> None:
        self._save_array(img, path)

    def _save_contour_hull(self, hull: np.ndarray | None, shape: tuple, path: Path) -> None:
        h, w = shape[:2]
        blank = np.zeros((h, w), dtype=np.uint8)
        if hull is not None and hull.size > 0:
            cv2.drawContours(blank, [hull], -1, 255, 2)
        self._save_array(blank, path)


def test_pipeline_visualization():
    config = Config(
        match_threshold=0.73,
        canny_threshold1=50.0,
        canny_threshold2=150.0,
    )

    pipeline = create_match_pipeline(
        loader=ImageLoader(),
        normalizer=PhotoNormalizer(config),
        matcher=ShapeMatcher(config),
        exporter=ResultExporter(),
    )
    visualizer = PipelineVisualizer(
        pipeline.matcher, pipeline.normalizer, "tests/output"
    )

    pairs = pipeline.load_directory("imgs/origin", "imgs/dummy")
    assert len(pairs) > 0

    origin_file, dummy_file, origin_img, dummy_img = pairs[0]
    result_name = dummy_file.replace(".", "_")
    visualizer.visualize_pipeline(origin_img, dummy_img, result_name)

    out = Path("tests/output")
    assert (out / f"{result_name}_origin_edges.png").exists()
    assert (out / f"{result_name}_dummy_edges.png").exists()


if __name__ == "__main__":
    test_pipeline_visualization()
