from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import glob
import os
import json

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

@app.get("/api/results")
def get_results():
    results = []
    json_files = glob.glob("output/*.json")
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                base_name = os.path.basename(jf).replace('_result.json', '')
                data['id'] = base_name
                data['result_image'] = f"/output/{base_name}_result.png"
                results.append(data)
        except Exception as e:
            continue
    # 類似度の降順でソート
    results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
