import pytest
from src.config import Config
from src.pipeline_factory import build_default_pipeline
from src.match_pipeline import MatchPipeline
from src.image_loader import ImageLoader
from src.photo_normalizer import PhotoNormalizer
from src.shape_matcher import ShapeMatcher
from src.result_exporter import ResultExporter


def test_build_default_pipeline():
    config = Config()
    pipeline = build_default_pipeline(config)
    
    assert isinstance(pipeline, MatchPipeline)
    assert isinstance(pipeline.loader, ImageLoader)
    assert isinstance(pipeline.normalizer, PhotoNormalizer)
    assert isinstance(pipeline.matcher, ShapeMatcher)
    assert isinstance(pipeline.exporter, ResultExporter)
    assert pipeline.normalizer.config == config
    assert pipeline.matcher.config == config
