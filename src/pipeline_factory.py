"""デフォルト構成のパイプライン生成（composition root 用）。"""
from src.config import Config
from src.image_loader import ImageLoader
from src.match_pipeline import MatchPipeline, create_match_pipeline
from src.photo_normalizer import PhotoNormalizer
from src.result_exporter import ResultExporter
from src.shape_matcher import ShapeMatcher


def build_default_pipeline(config: Config) -> MatchPipeline:
    return create_match_pipeline(
        loader=ImageLoader(),
        normalizer=PhotoNormalizer(config),
        matcher=ShapeMatcher(config),
        exporter=ResultExporter(),
    )
