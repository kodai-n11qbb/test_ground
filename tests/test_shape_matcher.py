import pytest
import numpy as np
import cv2
from src.config import Config
from src.shape_matcher import ShapeMatcher

@pytest.fixture
def config():
    return Config(match_threshold=0.8, canny_threshold1=50.0, canny_threshold2=150.0)

@pytest.fixture
def shape_matcher(config):
    return ShapeMatcher(config)

def create_shape_image(shape_type="square"):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    if shape_type == "square":
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    elif shape_type == "circle":
        cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
    return img

def test_match_shapes_identical(shape_matcher):
    img1 = create_shape_image("square")
    img2 = create_shape_image("square")

    result = shape_matcher.match_shapes(img1, img2, method="diff")
    assert result.is_match is True
    assert result.similarity_score > 0.99

def test_match_shapes_different(shape_matcher):
    img1 = create_shape_image("square")
    img2 = create_shape_image("circle")

    result = shape_matcher.match_shapes(img1, img2, method="diff")
    assert result.similarity_score < 1.0

def test_match_shapes_matchshapes_method(shape_matcher):
    square = create_shape_image("square")
    circle = create_shape_image("circle")

    identical = shape_matcher.match_shapes(square, square, method="matchshapes")
    different = shape_matcher.match_shapes(square, circle, method="matchshapes")

    assert identical.similarity_score > different.similarity_score
    assert 0.0 <= different.similarity_score <= 1.0

def test_hu_moments_calculation(shape_matcher):
    img = create_shape_image("square")
    processed = shape_matcher._preprocess(img)
    contours = shape_matcher._extract_contours(processed)

    hu = shape_matcher._calculate_hu_moments_from_contours(contours)
    assert hu is not None
    assert len(hu) == 7

def test_compare_hu_moments_diff(shape_matcher):
    hu1 = np.array([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
    hu2 = np.array([1.1, 0.11, 0.011, 0.0011, 0.00011, 0.000011, 0.0000011])

    sim = shape_matcher._compare_hu_moments(hu1, hu2)
    assert 0.0 <= sim <= 1.0

def test_compare_contours_match_shapes(shape_matcher):
    img = create_shape_image("square")
    processed = shape_matcher._preprocess(img)
    contours = shape_matcher._extract_contours(processed)

    sim = shape_matcher._compare_contours_match_shapes(contours, contours)
    assert sim > 0.9
