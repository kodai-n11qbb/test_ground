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
        元画像とダミー画像をブレンドし、はみ出している（差分）箇所を赤くハイライトして出力する。
        """
        if result.origin_img is not None and result.dummy_img is not None:
            # 50/50ブレンドで全体構造を薄く見せる
            blend = cv2.addWeighted(result.origin_img, 0.5, result.dummy_img, 0.5, 0)
            
            # originの輪郭を緑色で描画
            if result.origin_contours is not None:
                cv2.drawContours(blend, result.origin_contours, -1, (0, 255, 0), 2)
                
            # dummyの輪郭を赤色で描画
            if result.dummy_contours is not None:
                cv2.drawContours(blend, result.dummy_contours, -1, (0, 0, 255), 2)
                
            img = blend
        else:
            # フォールバック
            img = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # ディレクトリ作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存 (日本語パス対応のためimencodeを使用)
        cv2.imencode(".png", img)[1].tofile(output_path)
    
    def export_json(self, result: MatchResult, output_path: str) -> None:
        """
        結果をJSONファイルとして出力する。
        """
        data = {
            "similarity_score": result.similarity_score,
            "is_match": result.is_match,
            "origin_path": result.origin_path,
            "dummy_path": result.dummy_path,
            "photo_normalized": result.photo_normalized,
            "hu_moments_origin": result.hu_moments_origin.tolist() if result.hu_moments_origin is not None else None,
            "hu_moments_dummy": result.hu_moments_dummy.tolist() if result.hu_moments_dummy is not None else None
        }
        
        # ディレクトリ作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
