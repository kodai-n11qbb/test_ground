import cv2
import json
import os
import numpy as np
from typing import Optional
from .models import MatchResult


class ResultExporter:
    def export_image(self, result: MatchResult, output_path: str) -> None:
        """
        結果を画像ファイルとして出力する。
        現在は簡易的に、類似度をテキストとして描画した画像を出力。
        """
        # 黒い画像を作成
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # テキスト描画
        status = "MATCH" if result.is_match else "NO MATCH"
        color = (0, 255, 0) if result.is_match else (0, 0, 255)
        
        cv2.putText(img, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Similarity: {result.similarity_score:.3f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ディレクトリ作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存
        cv2.imwrite(output_path, img)
    
    def export_json(self, result: MatchResult, output_path: str) -> None:
        """
        結果をJSONファイルとして出力する。
        """
        data = {
            "similarity_score": result.similarity_score,
            "is_match": result.is_match,
            "origin_path": result.origin_path,
            "dummy_path": result.dummy_path,
            "hu_moments_origin": result.hu_moments_origin.tolist() if result.hu_moments_origin is not None else None,
            "hu_moments_dummy": result.hu_moments_dummy.tolist() if result.hu_moments_dummy is not None else None
        }
        
        # ディレクトリ作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
