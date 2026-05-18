import pytest
import cv2
import numpy as np
import os
from src.image_loader import ImageLoader

@pytest.fixture
def image_loader():
    return ImageLoader()

@pytest.fixture
def create_dummy_image(tmp_path):
    def _create_image(filename):
        path = os.path.join(tmp_path, filename)
        # Create a simple white image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Since cv2.imwrite might not handle Japanese characters well in all systems
        # we will use numpy to write it as done in ImageLoader reading.
        # But actually, cv2.imencode -> tofile works.
        is_success, im_buf_arr = cv2.imencode(".png", img)
        im_buf_arr.tofile(path)
        return path
    return _create_image

def test_load_image_pair_success(image_loader, create_dummy_image):
    origin_path = create_dummy_image("origin_test.png")
    dummy_path = create_dummy_image("dummy_test.png")
    
    origin, dummy = image_loader.load_image_pair(origin_path, dummy_path)
    assert origin is not None
    assert dummy is not None
    assert origin.shape == (100, 100, 3)
    assert dummy.shape == (100, 100, 3)

def test_load_image_pair_not_found(image_loader):
    with pytest.raises(FileNotFoundError):
        image_loader.load_image_pair("non_existent_1.png", "non_existent_2.png")

def test_load_directory(image_loader, tmp_path, create_dummy_image):
    origin_dir = os.path.join(tmp_path, "origin")
    dummy_dir = os.path.join(tmp_path, "dummy")
    os.makedirs(origin_dir)
    os.makedirs(dummy_dir)
    
    # Create matching pairs based on Japanese/English keywords
    # origin
    create_dummy_image(os.path.join("origin", "fileA(ドトール).png"))
    create_dummy_image(os.path.join("origin", "fileB（吉野家）.png"))
    
    # dummy
    create_dummy_image(os.path.join("dummy", "dummyA(ドトール)入れ替え.png"))
    create_dummy_image(os.path.join("dummy", "dummyB（吉野家）入れ替え.png"))
    create_dummy_image(os.path.join("dummy", "unrelated.png"))
    
    pairs = image_loader.load_directory(origin_dir, dummy_dir)
    
    assert len(pairs) == 2
    origin_names = [p[0] for p in pairs]
    dummy_names = [p[1] for p in pairs]
    assert "fileA(ドトール).png" in origin_names
    assert "dummyA(ドトール)入れ替え.png" in dummy_names
