import json
import os
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture
def client():
    return TestClient(app)


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
