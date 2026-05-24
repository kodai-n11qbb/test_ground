import os
import tempfile
import pytest
from src.config import Config

def test_config_default_values():
    config = Config()
    assert config.hsv_lower1 == [90, 40, 30]
    assert config.hsv_upper1 == [140, 255, 255]
    assert config.hsv_lower2 == [100, 20, 20]
    assert config.hsv_upper2 == [150, 255, 180]
    assert config.match_threshold == 0.70

def test_config_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        
        # カスタムパラメータでConfigを作成
        config = Config(
            match_threshold=0.85,
            hsv_lower1=[80, 50, 40],
            hsv_upper1=[130, 240, 240]
        )
        
        # 保存
        config.save_to_json(config_path)
        assert os.path.exists(config_path)
        
        # 読み込み
        loaded_config = Config.load_from_json(config_path)
        
        assert loaded_config.match_threshold == 0.85
        assert loaded_config.hsv_lower1 == [80, 50, 40]
        assert loaded_config.hsv_upper1 == [130, 240, 240]
        
        # 他のデフォルト値が維持されているか確認
        assert loaded_config.hsv_lower2 == [100, 20, 20]

def test_config_load_non_existent():
    # 存在しないパスからのロードはデフォルト値になる
    config = Config.load_from_json("non_existent_config_file_name_12345.json")
    assert config.match_threshold == 0.70
    assert config.hsv_lower1 == [90, 40, 30]
