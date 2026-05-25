from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import glob
import os
import json
import numpy as np
from urllib.parse import quote, unquote

from src.config import Config
from src.pipeline_factory import build_default_pipeline

app = FastAPI(title="画像差分検出 Viewer")

# ディレクトリが存在しない場合の対策
os.makedirs("output", exist_ok=True)
os.makedirs("output/preview", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount directories
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")

# グローバル設定とパイプラインの初期化 (Dependency Injection)
config = Config.load_from_json()
pipeline = build_default_pipeline(config)


class TunerParams(BaseModel):
    hsv_lower1: list[int]
    hsv_upper1: list[int]
    hsv_lower2: list[int]
    hsv_upper2: list[int]
    match_threshold: float


@app.get("/")
def read_index():
    return FileResponse("static/index.html")


@app.get("/tuner")
def read_tuner():
    return FileResponse("static/tuner.html")


@app.get("/api/config")
def get_config():
    from dataclasses import asdict
    return asdict(config)


@app.post("/api/tuner/preview")
def tuner_preview(params: TunerParams):
    preview_config = Config(
        hsv_lower1=params.hsv_lower1,
        hsv_upper1=params.hsv_upper1,
        hsv_lower2=params.hsv_lower2,
        hsv_upper2=params.hsv_upper2,
        match_threshold=params.match_threshold,
        output_dir="output/preview",
        origin_dir=config.origin_dir,
        dummy_dir=config.dummy_dir
    )
    
    # プレビュー用の別pipelineを構築（DI）
    preview_pipeline = build_default_pipeline(preview_config)
    
    test_images = [
        ("実写直近(ドトール).jpg", f"{config.dummy_dir}/実写直近(ドトール).jpg", f"{config.origin_dir}/POLITEC(ドトール).png"),
        ("実写遠近(ドトール).jpg", f"{config.dummy_dir}/実写遠近(ドトール).jpg", f"{config.origin_dir}/POLITEC(ドトール).png"),
        ("dummy(ドトール)入れ替えのみ_00_TEOPCIL.jpg", f"{config.dummy_dir}/dummy(ドトール)入れ替えのみ_00_TEOPCIL.jpg", f"{config.origin_dir}/POLITEC(ドトール).png")
    ]
    
    results = {}
    import cv2
    
    for filename, dummy_path, origin_path in test_images:
        if not os.path.exists(origin_path) or not os.path.exists(dummy_path):
            continue
        origin_img = cv2.imdecode(np.fromfile(origin_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        dummy_img = cv2.imdecode(np.fromfile(dummy_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        needs_norm = preview_pipeline.normalizer.needs_normalization(dummy_img, origin_img)
        if needs_norm:
            flat_img = preview_pipeline.normalizer.normalize(dummy_img, origin_img)
        else:
            flat_img = dummy_img
            
        res = preview_pipeline.matcher.match_shapes(origin_img, flat_img)
        res.photo_normalized = needs_norm
        
        out_name = filename.replace('.', '_')
        cv2.imencode(".png", flat_img)[1].tofile(f"output/preview/{out_name}_flat.png")
        preview_pipeline.exporter.export_image(res, f"output/preview/{out_name}_overlay.png")
        
        hsv_mask = preview_pipeline.normalizer._blue_mask_hsv(flat_img)
        cv2.imencode(".png", hsv_mask)[1].tofile(f"output/preview/{out_name}_hsv_mask.png")
        
        results[filename] = {"score": res.similarity_score, "is_match": res.is_match}
        
    return {"status": "success", "results": results}


@app.post("/api/save-config")
def save_config(params: TunerParams):
    global config, pipeline
    
    # 1. グローバル設定を更新
    config.hsv_lower1 = params.hsv_lower1
    config.hsv_upper1 = params.hsv_upper1
    config.hsv_lower2 = params.hsv_lower2
    config.hsv_upper2 = params.hsv_upper2
    config.match_threshold = params.match_threshold
    
    # 2. jsonに保存
    config.save_to_json()
    
    # 3. パイプラインを再構築（DI）
    pipeline = build_default_pipeline(config)
    
    # 4. 全画像ペアに対して再判定を実行（同期処理）
    pairs = pipeline.load_directory(config.origin_dir, config.dummy_dir)
    for origin_file, dummy_file, origin_img, dummy_img in pairs:
        result = pipeline.process_pair(origin_img, dummy_img, method=config.match_method)
        result.origin_path = f"{config.origin_dir}/{origin_file}"
        result.dummy_path = f"{config.dummy_dir}/{dummy_file}"
        
        output_name = dummy_file.replace('.', '_')
        image_path = f"output/{output_name}_result.png"
        json_path = f"output/{output_name}_result.json"
        
        pipeline.export_result(result, image_path, json_path)
        
    return {"status": "success", "message": "Configuration saved and pipeline re-run completed."}


def _result_image_url(base_name: str) -> str:
    """日本語ファイル名をブラウザが取得できるよう API 経由の URL にする。"""
    return "/api/result-image/" + quote(base_name, safe="")


@app.get("/api/result-image/{result_id}")
def get_result_image(result_id: str):
    base_name = os.path.basename(unquote(result_id))
    if not base_name or ".." in base_name:
        raise HTTPException(status_code=400, detail="Invalid result id")
    path = os.path.join("output", f"{base_name}_result.png")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/results")
def get_results():
    results = []
    json_files = glob.glob("output/*_result.json")
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                base_name = os.path.basename(jf).replace('_result.json', '')
                data['id'] = base_name
                data['result_image'] = _result_image_url(base_name)
                results.append(data)
        except Exception:
            continue
    results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
