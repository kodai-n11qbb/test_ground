import cv2
import numpy as np
from typing import Tuple, List
import os


class ImageLoader:
    def load_image_pair(self, origin_path: str, dummy_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        指定されたパスの画像ペアを読み込む。
        """
        # パスを正規化（Windows対応）
        origin_path = os.path.normpath(origin_path)
        dummy_path = os.path.normpath(dummy_path)
        
        # cv2.imreadは日本語パスに対応していないため、np.fromfileを使用
        try:
            with open(origin_path, 'rb') as f:
                origin_bytes = np.fromfile(f, dtype=np.uint8)
            origin = cv2.imdecode(origin_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise FileNotFoundError(f"Origin image not found: {origin_path}, error: {e}")
        
        try:
            with open(dummy_path, 'rb') as f:
                dummy_bytes = np.fromfile(f, dtype=np.uint8)
            dummy = cv2.imdecode(dummy_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise FileNotFoundError(f"Dummy image not found: {dummy_path}, error: {e}")
        
        if origin is None:
            raise FileNotFoundError(f"Origin image not found: {origin_path}")
        if dummy is None:
            raise FileNotFoundError(f"Dummy image not found: {dummy_path}")
        
        return origin, dummy
    
    def load_directory(self, origin_dir: str, dummy_dir: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        ディレクトリ内の全画像ペアを読み込む。
        カッコ内のキーワードでマッチングする。
        """
        origin_files = sorted(os.listdir(origin_dir))
        dummy_files = sorted(os.listdir(dummy_dir))
        
        # キーワード抽出（カッコ内のテキスト）
        def extract_keyword(filename):
            import re
            # 全角カッコまたは半角カッコ内のテキストを抽出
            match = re.search(r'\uff08([^\uff09]+)\uff09', filename)
            if match:
                return match.group(1)
            match = re.search(r'\(([^)]+)\)', filename)
            if match:
                return match.group(1)
            return ""
        
        pairs = []
        for origin_file in origin_files:
            origin_keyword = extract_keyword(origin_file)
            if not origin_keyword:
                continue
            
            for dummy_file in dummy_files:
                dummy_keyword = extract_keyword(dummy_file)
                if not dummy_keyword:
                    continue
                
                # キーワードが一致する場合
                if origin_keyword == dummy_keyword:
                    origin_path = os.path.join(origin_dir, origin_file)
                    dummy_path = os.path.join(dummy_dir, dummy_file)
                    origin, dummy = self.load_image_pair(origin_path, dummy_path)
                    pairs.append((dummy_file, origin, dummy))
        
        return pairs
