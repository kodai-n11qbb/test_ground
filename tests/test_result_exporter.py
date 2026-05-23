import pytest
import os
import json
import numpy as np
from src.models import MatchResult
from src.result_exporter import ResultExporter

@pytest.fixture
def result_exporter():
    return ResultExporter()

@pytest.fixture
def dummy_result():
    return MatchResult(
        similarity_score=0.95,
        is_match=True,
        origin_path="origin/path/test.png",
        dummy_path="dummy/path/test.png",
        hu_moments_origin=np.zeros(7),
        hu_moments_dummy=np.zeros(7),
        photo_normalized=True,
    )

def test_export_image(result_exporter, dummy_result, tmp_path):
    output_path = os.path.join(tmp_path, "output.png")
    
    result_exporter.export_image(dummy_result, output_path)
    
    assert os.path.exists(output_path)
    # 簡易的にファイルサイズが0より大きいか確認
    assert os.path.getsize(output_path) > 0

def test_export_json(result_exporter, dummy_result, tmp_path):
    output_path = os.path.join(tmp_path, "output.json")
    
    result_exporter.export_json(dummy_result, output_path)
    
    assert os.path.exists(output_path)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    assert data["similarity_score"] == 0.95
    assert data["is_match"] is True
    assert data["origin_path"] == "origin/path/test.png"
    assert data["dummy_path"] == "dummy/path/test.png"
    assert data["photo_normalized"] is True
    assert len(data["hu_moments_origin"]) == 7


def test_export_image_with_contours(result_exporter, dummy_result, tmp_path):
    output_path = os.path.join(tmp_path, "output_contours.png")
    
    # Set mock images
    dummy_result.origin_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    dummy_result.dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    
    # Set mock contours
    origin_cnt = [np.array([[[10, 10]], [[10, 90]], [[90, 90]], [[90, 10]]], dtype=np.int32)]
    dummy_cnt = [np.array([[[15, 15]], [[15, 85]], [[85, 85]], [[85, 15]]], dtype=np.int32)]
    
    dummy_result.origin_contours = origin_cnt
    dummy_result.dummy_contours = dummy_cnt
    
    result_exporter.export_image(dummy_result, output_path)
    
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

