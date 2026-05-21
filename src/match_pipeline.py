"""画像ペアの読み込み・正規化・照合・出力をつなぐパイプライン（DI）。"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .image_loader import ImageLoader
from .models import MatchResult
from .photo_normalizer import PhotoNormalizer
from .result_exporter import ResultExporter
from .shape_matcher import ShapeMatcher


class MatchPipeline:
    def __init__(
        self,
        loader: ImageLoader,
        normalizer: PhotoNormalizer,
        matcher: ShapeMatcher,
        exporter: ResultExporter,
    ) -> None:
        self._loader = loader
        self._normalizer = normalizer
        self._matcher = matcher
        self._exporter = exporter

    @property
    def loader(self) -> ImageLoader:
        return self._loader

    @property
    def normalizer(self) -> PhotoNormalizer:
        return self._normalizer

    @property
    def matcher(self) -> ShapeMatcher:
        return self._matcher

    @property
    def exporter(self) -> ResultExporter:
        return self._exporter

    def prepare_dummy_for_match(
        self, origin_img: np.ndarray, dummy_img: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        if self._normalizer.needs_normalization(dummy_img, origin_img):
            return self._normalizer.normalize(dummy_img, origin_img), True
        return dummy_img, False

    def process_pair(
        self,
        origin_img: np.ndarray,
        dummy_img: np.ndarray,
        method: str = "diff",
    ) -> MatchResult:
        dummy_for_match, photo_normalized = self.prepare_dummy_for_match(origin_img, dummy_img)
        result = self._matcher.match_shapes(origin_img, dummy_for_match, method=method)
        result.photo_normalized = photo_normalized
        return result

    def export_result(self, result: MatchResult, image_path: str, json_path: str) -> None:
        self._exporter.export_image(result, image_path)
        self._exporter.export_json(result, json_path)

    def load_directory(
        self, origin_dir: str, dummy_dir: str
    ) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
        return self._loader.load_directory(origin_dir, dummy_dir)


def create_match_pipeline(
    loader: ImageLoader,
    normalizer: PhotoNormalizer,
    matcher: ShapeMatcher,
    exporter: ResultExporter,
) -> MatchPipeline:
    """依存を外から受け取りパイプラインを組み立てる。"""
    return MatchPipeline(loader, normalizer, matcher, exporter)
