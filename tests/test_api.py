import json
import os
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture
def client():
    return TestClient(app)


def save_image_robust(path, img):
    import cv2
    ext = os.path.splitext(path)[1]
    result, nparr = cv2.imencode(ext, img)
    if result:
        with open(path, mode='wb') as f:
            nparr.tofile(f)


def test_result_image_serves_unicode_filename(client, tmp_path, monkeypatch):
    """日本語 id の結果画像を API 経由で取得できる。"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    base = "実写直近(ドトール)_jpg"
    png_path = output_dir / f"{base}_result.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    json_path = output_dir / f"{base}_result.json"
    json_path.write_text(
        json.dumps(
            {
                "similarity_score": 0.73,
                "is_match": False,
                "origin_path": "imgs/origin/x.png",
                "dummy_path": "imgs/dummy/x.jpg",
                "photo_normalized": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    res = client.get("/api/results")
    assert res.status_code == 200
    items = res.json()["results"]
    assert len(items) == 1
    assert items[0]["result_image"].startswith("/api/result-image/")

    img_res = client.get(items[0]["result_image"])
    assert img_res.status_code == 200
    assert img_res.headers["content-type"] == "image/png"


def test_result_image_not_found(client):
    res = client.get("/api/result-image/" + quote("missing_id_xyz", safe=""))
    assert res.status_code == 404


def test_tuner_get_page(client):
    res = client.get("/tuner")
    assert res.status_code == 200
    assert b"HSV Mask Parameter Tuner" in res.content


def test_api_get_config(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    cfg = res.json()
    assert "hsv_lower1" in cfg
    assert "match_threshold" in cfg


def test_api_tuner_preview(client, tmp_path, monkeypatch):
    origin_dir = tmp_path / "origin"
    dummy_dir = tmp_path / "dummy"
    output_dir = tmp_path / "output"
    origin_dir.mkdir()
    dummy_dir.mkdir()
    output_dir.mkdir()

    # Create dummy images with expected names
    import numpy as np
    img = np.zeros((100, 440, 3), dtype=np.uint8)
    save_image_robust(str(origin_dir / "POLITEC(ドトール).png"), img)
    save_image_robust(str(dummy_dir / "実写直近(ドトール).jpg"), img)
    save_image_robust(str(dummy_dir / "実写遠近(ドトール).jpg"), img)
    save_image_robust(str(dummy_dir / "dummy(ドトール)入れ替えのみ_00_TEOPCIL.jpg"), img)

    import api
    monkeypatch.setattr(api.config, "origin_dir", str(origin_dir))
    monkeypatch.setattr(api.config, "dummy_dir", str(dummy_dir))
    monkeypatch.setattr(api.config, "output_dir", str(output_dir))

    payload = {
        "hsv_lower1": [90, 40, 30],
        "hsv_upper1": [140, 255, 255],
        "hsv_lower2": [100, 20, 20],
        "hsv_upper2": [150, 255, 180],
        "match_threshold": 0.70
    }
    
    # Generate actual local paths in tmp_path instead of output/
    os.makedirs(tmp_path / "output/preview", exist_ok=True)
    monkeypatch.chdir(tmp_path)

    res = client.post("/api/tuner/preview", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "実写直近(ドトール).jpg" in data["results"]


def test_api_tuner_preview_missing_images(client, tmp_path, monkeypatch):
    """テスト画像が一部または全部存在しない場合でも、APIが500にならず200を返し、空の結果を返す。"""
    origin_dir = tmp_path / "origin"
    dummy_dir = tmp_path / "dummy"
    output_dir = tmp_path / "output"
    origin_dir.mkdir()
    dummy_dir.mkdir()
    output_dir.mkdir()

    # 画像を意図的に作成しない（テスト画像が欠損している状態）

    import api
    monkeypatch.setattr(api.config, "origin_dir", str(origin_dir))
    monkeypatch.setattr(api.config, "dummy_dir", str(dummy_dir))
    monkeypatch.setattr(api.config, "output_dir", str(output_dir))

    payload = {
        "hsv_lower1": [90, 40, 30],
        "hsv_upper1": [140, 255, 255],
        "hsv_lower2": [100, 20, 20],
        "hsv_upper2": [150, 255, 180],
        "match_threshold": 0.70
    }
    
    os.makedirs(tmp_path / "output/preview", exist_ok=True)
    monkeypatch.chdir(tmp_path)

    res = client.post("/api/tuner/preview", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    # 結果が空であることを検証
    assert len(data["results"]) == 0


def test_api_save_config(client, tmp_path, monkeypatch):
    origin_dir = tmp_path / "origin"
    dummy_dir = tmp_path / "dummy"
    output_dir = tmp_path / "output"
    origin_dir.mkdir()
    dummy_dir.mkdir()
    output_dir.mkdir()

    import numpy as np
    img = np.zeros((100, 440, 3), dtype=np.uint8)
    save_image_robust(str(origin_dir / "テスト(ドトール).png"), img)
    save_image_robust(str(dummy_dir / "テスト(ドトール).jpg"), img)

    import api
    monkeypatch.setattr(api.config, "origin_dir", str(origin_dir))
    monkeypatch.setattr(api.config, "dummy_dir", str(dummy_dir))
    monkeypatch.setattr(api.config, "output_dir", str(output_dir))

    config_json_path = str(tmp_path / "config.json")
    original_save = api.config.save_to_json
    monkeypatch.setattr(api.config, "save_to_json", lambda *args, **kwargs: original_save(config_json_path))
    monkeypatch.chdir(tmp_path)

    payload = {
        "hsv_lower1": [80, 50, 40],
        "hsv_upper1": [130, 240, 240],
        "hsv_lower2": [95, 25, 25],
        "hsv_upper2": [145, 245, 175],
        "match_threshold": 0.85
    }
    res = client.post("/api/save-config", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"

    assert os.path.exists(config_json_path)
    loaded = api.Config.load_from_json(config_json_path)
    assert loaded.match_threshold == 0.85
    assert loaded.hsv_lower1 == [80, 50, 40]
