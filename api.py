from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import glob
import os
import json
from urllib.parse import quote, unquote

app = FastAPI(title="画像差分検出 Viewer")

# ディレクトリが存在しない場合の対策
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount directories
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

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
