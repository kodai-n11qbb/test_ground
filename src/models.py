from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MatchResult:
    similarity_score: float  # 0.0 - 1.0
    is_match: bool  # 閾値による判定
    origin_path: str
    dummy_path: str
    hu_moments_origin: Optional[np.ndarray] = None
    hu_moments_dummy: Optional[np.ndarray] = None
