from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
from pathlib import Path

ROOT = Path("/Users/abekoudai/Desktop/test_ground")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sandbox"))

from sandbox.test_hsv_tuning import run_tuning_pipeline

app = FastAPI(title="HSV Mask Tuner API (Sandbox)")

# Mount output directory generated in sandbox/output
out_path = ROOT / "sandbox" / "output"
out_path.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory=str(out_path)), name="output")

class PipelineParams(BaseModel):
    lower1: str
    upper1: str
    lower2: str
    upper2: str
    threshold: float

@app.get("/")
def read_index():
    return FileResponse(str(ROOT / "sandbox" / "index.html"))

@app.post("/api/run")
def run_pipeline(params: PipelineParams):
    try:
        def parse_csv(s):
            return [int(x) for x in s.split(',')]
            
        hsv_lower1 = parse_csv(params.lower1)
        hsv_upper1 = parse_csv(params.upper1)
        hsv_lower2 = parse_csv(params.lower2)
        hsv_upper2 = parse_csv(params.upper2)
        
        results = run_tuning_pipeline(
            hsv_lower1, hsv_upper1,
            hsv_lower2, hsv_upper2,
            params.threshold
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_sandbox:app", host="127.0.0.1", port=8002, reload=True)
