from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
import base64
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sandbox"))

from sandbox.test_hsv_tuning import run_tuning_pipeline, run_single_live_frame, robust_imread

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

class LiveParams(BaseModel):
    image: str
    lower1: str
    upper1: str
    lower2: str
    upper2: str
    threshold: float
    origin_name: str

@app.post("/api/live")
def run_live_pipeline(params: LiveParams):
    try:
        def parse_csv(s):
            return [int(x) for x in s.split(',')]
            
        hsv_lower1 = parse_csv(params.lower1)
        hsv_upper1 = parse_csv(params.upper1)
        hsv_lower2 = parse_csv(params.lower2)
        hsv_upper2 = parse_csv(params.upper2)
        
        # Decode base64 image
        header, encoded = params.image.split(",", 1) if "," in params.image else ("", params.image)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        # Load origin (reference) image
        origin_path = ROOT / "imgs" / "origin" / params.origin_name
        if not origin_path.exists():
            raise HTTPException(status_code=404, detail=f"Reference image {params.origin_name} not found")
        origin_img = robust_imread(origin_path)
        if origin_img is None:
            raise HTTPException(status_code=400, detail="Could not load reference image")
            
        # Run pipeline
        results = run_single_live_frame(
            frame, origin_img,
            hsv_lower1, hsv_upper1,
            hsv_lower2, hsv_upper2,
            params.threshold
        )
        
        # Helper to convert cv2 image to base64
        def to_b64(img):
            _, buffer = cv2.imencode(".png", img)
            b64_str = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"
            
        return {
            "status": "success",
            "similarity_score": results["similarity_score"],
            "is_match": results["is_match"],
            "hsv_mask": to_b64(results["hsv_mask"]),
            "flat_img": to_b64(results["flat_img"]),
            "overlay_img": to_b64(results["overlay_img"])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_sandbox:app", host="127.0.0.1", port=8002, reload=True)
