import cv2
import numpy as np
import os
import sys
from pathlib import Path

def check_origin_box():
    ROOT = Path("/Users/abekoudai/Desktop/test_ground")
    img = cv2.imread(str(ROOT / "imgs/origin/POLITEC(ドトール).png"))
    h, w = img.shape[:2]
    print(f"Origin Size: {w}x{h}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        print(f"Bounding Box: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
        print(f"Width={x1-x0}, Height={y1-y0}")
        print(f"Aspect Ratio (W/H): {(x1-x0)/(y1-y0):.3f}")
        
if __name__ == "__main__":
    check_origin_box()
