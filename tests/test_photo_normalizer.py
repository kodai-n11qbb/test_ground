import cv2
import numpy as np
import pytest

from src.config import Config
from src.photo_normalizer import PhotoNormalizer, CAD_BLUE_BGR, CAD_WHITE_BGR


@pytest.fixture
def normalizer():
    return PhotoNormalizer(Config(photo_normalize_enabled=True))


def _solid_blue_photo(h: int = 400, w: int = 600) -> np.ndarray:
    """木目っぽい背景 + 中央に青矩形（実写の簡易模倣）。"""
    img = np.full((h, w, 3), (200, 180, 160), dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (w - 80, h - 120), CAD_BLUE_BGR, -1)
    return img


def _origin_cad(h: int = 169, w: int = 743) -> np.ndarray:
    img = np.full((h, w, 3), CAD_WHITE_BGR, dtype=np.uint8)
    cv2.rectangle(img, (50, 40), (w - 50, h - 40), CAD_BLUE_BGR, -1)
    return img


def test_needs_normalization_large_photo(normalizer):
    origin = _origin_cad()
    photo = _solid_blue_photo(1100, 1400)
    assert normalizer.needs_normalization(photo, origin) is True


def test_needs_normalization_skip_same_scale(normalizer):
    origin = _origin_cad()
    dummy = _origin_cad()
    assert normalizer.needs_normalization(dummy, origin) is False


def test_needs_normalization_disabled():
    n = PhotoNormalizer(Config(photo_normalize_enabled=False))
    photo = _solid_blue_photo(1100, 1400)
    origin = _origin_cad()
    assert n.needs_normalization(photo, origin) is False


def test_normalize_output_shape_and_colors(normalizer):
    origin = _origin_cad()
    photo = _solid_blue_photo()
    out = normalizer.normalize(photo, origin)
    assert out.shape == origin.shape
    assert np.any(np.all(out == CAD_BLUE_BGR, axis=2))
    pixels = out.reshape(-1, 3)
    unique = {tuple(p) for p in pixels}
    assert unique.issubset({CAD_BLUE_BGR, CAD_WHITE_BGR})


def test_normalize_improves_pixel_diff_vs_raw_resize(normalizer):
    """正規化後の absdiff は単純リサイズより小さい（ドメイン揃えの効果）。"""
    origin = _origin_cad()
    photo = _solid_blue_photo()
    norm = normalizer.normalize(photo, origin)
    raw = cv2.resize(photo, (origin.shape[1], origin.shape[0]))
    diff_norm = cv2.absdiff(origin, norm)
    diff_raw = cv2.absdiff(origin, raw)
    assert int((diff_norm > 30).sum()) < int((diff_raw > 30).sum())


def test_order_points(normalizer):
    # Unsorted points
    pts = np.array([[100, 100], [0, 100], [100, 0], [0, 0]], dtype=np.float32)
    ordered = normalizer._order_points(pts)
    # Expected order: Top-Left [0,0], Top-Right [100,0], Bottom-Right [100,100], Bottom-Left [0,100]
    expected = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert np.allclose(ordered, expected)


def test_detect_four_corners(normalizer):
    # Create a mask with a rectangle
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (30, 30), (170, 170), 255, -1)
    
    corners = normalizer._detect_four_corners(mask)
    assert corners is not None
    assert len(corners) == 4
    # Check that corners are approximately the rectangle boundaries
    # Sorted corners should match the expected rect points roughly
    ordered = normalizer._order_points(corners)
    # Expected corners: (30, 30), (170, 30), (170, 170), (30, 170)
    for p, exp in zip(ordered, [[30, 30], [170, 30], [170, 170], [30, 170]]):
        assert np.linalg.norm(p - exp) < 5.0


def test_detect_four_corners_extremum(normalizer):
    normalizer.config.photo_corner_detection_method = "extremum"
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (30, 30), (170, 170), 255, -1)
    
    corners = normalizer._detect_four_corners(mask)
    assert corners is not None
    assert len(corners) == 4
    ordered = normalizer._order_points(corners)
    for p, exp in zip(ordered, [[30, 30], [170, 30], [170, 170], [30, 170]]):
        assert np.linalg.norm(p - exp) < 5.0


def test_detect_four_corners_rotated(normalizer):
    normalizer.config.photo_corner_detection_method = "rotated"
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (30, 30), (170, 170), 255, -1)
    
    corners = normalizer._detect_four_corners(mask)
    assert corners is not None
    assert len(corners) == 4


def test_normalize_preserves_aspect_ratio(normalizer):
    origin = _origin_cad(h=169, w=743)
    photo = _solid_blue_photo(h=400, w=600)
    out = normalizer.normalize(photo, origin)
    assert out.shape == origin.shape

def test_normalize_stretch_alignment_mode():
    config = Config(photo_normalize_enabled=True, photo_alignment_mode="stretch")
    normalizer = PhotoNormalizer(config)
    origin = _origin_cad(h=100, w=200)
    photo = _solid_blue_photo(h=400, w=800)
    out = normalizer.normalize(photo, origin)
    assert out.shape == origin.shape


