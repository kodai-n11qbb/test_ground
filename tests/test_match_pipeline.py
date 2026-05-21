import cv2
import numpy as np
import pytest

from src.config import Config
from src.image_loader import ImageLoader
from src.match_pipeline import MatchPipeline, create_match_pipeline
from src.photo_normalizer import CAD_BLUE_BGR, CAD_WHITE_BGR, PhotoNormalizer
from src.result_exporter import ResultExporter
from src.shape_matcher import ShapeMatcher


@pytest.fixture
def config():
    return Config(match_threshold=0.8, photo_size_ratio_threshold=1.5)


@pytest.fixture
def pipeline(config):
    return create_match_pipeline(
        loader=ImageLoader(),
        normalizer=PhotoNormalizer(config),
        matcher=ShapeMatcher(config),
        exporter=ResultExporter(),
    )


def _origin_cad(h: int = 169, w: int = 743) -> np.ndarray:
    img = np.full((h, w, 3), CAD_WHITE_BGR, dtype=np.uint8)
    cv2.rectangle(img, (50, 40), (w - 50, h - 40), CAD_BLUE_BGR, -1)
    return img


def _large_photo(h: int = 1100, w: int = 1400) -> np.ndarray:
    img = np.full((h, w, 3), (200, 180, 160), dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (w - 80, h - 120), CAD_BLUE_BGR, -1)
    return img


def test_create_match_pipeline_injects_dependencies(pipeline):
    assert isinstance(pipeline.loader, ImageLoader)
    assert isinstance(pipeline.normalizer, PhotoNormalizer)
    assert isinstance(pipeline.matcher, ShapeMatcher)
    assert isinstance(pipeline.exporter, ResultExporter)


def test_process_pair_normalizes_large_photo(pipeline):
    origin = _origin_cad()
    photo = _large_photo()

    result = pipeline.process_pair(origin, photo, method="diff")

    assert result.photo_normalized is True
    assert result.dummy_img.shape == origin.shape


def test_process_pair_skips_normalization_for_cad_scale_dummy(pipeline):
    origin = _origin_cad()
    dummy = _origin_cad()

    result = pipeline.process_pair(origin, dummy, method="diff")

    assert result.photo_normalized is False


def test_prepare_dummy_for_match(pipeline):
    origin = _origin_cad()
    photo = _large_photo()
    prepared, normalized = pipeline.prepare_dummy_for_match(origin, photo)
    assert normalized is True
    assert prepared.shape == origin.shape
