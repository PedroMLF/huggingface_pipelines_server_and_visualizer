import numpy as np
import pytest
from omegaconf import OmegaConf

from src.pipelines.base_pipeline import BasePipeline

MODEL_NAME = "sshleifer/tiny-distilbert-base-cased"
PIPELINE_NAME = "TokenClassificationPipeline"


class TestBaselinePipeline:
    def setup_class(cls):
        config = OmegaConf.create({"pipeline": PIPELINE_NAME, "model": MODEL_NAME})
        cls.pipeline = BasePipeline(config)

    def teardown_class(cls):
        pass

    def test_model_is_correct(self):
        assert self.pipeline.model == MODEL_NAME

    def test_hf_pipeline_is_correct(self):
        assert self.pipeline.hf_pipeline == PIPELINE_NAME

    def test_prefix_is_correct(self):
        assert self.pipeline.prefix == "##"

    def test_call_raises_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            self.pipeline("x")

    def test_tokenize_text(self):
        assert self.pipeline.tokenize_text("Example sentence!") == ["Example", "sentence", "!"]
        assert self.pipeline.tokenize_text("António Nunes!") == ["António", "Nunes", "!"]
        assert self.pipeline.tokenize_text("sgoiw dfdfer") == ["sgoiw", "dfdfer"]

    def test_map_all_np_keys(self):
        out = self.pipeline._map_all_np_keys(
            [
                {"x": "test1", "y": np.int64(42), "z": np.float64(42.0)},
                {"x": "test2", "y": np.int64(22), "z": np.float64(22.0)},
                {"x": "test3", "y": np.int64(12), "z": np.float64(12.0)},
            ]
        )
        assert out[0]["x"] == "test1"
        assert out[0]["y"] == 42
        assert out[0]["z"] == 42.0
        assert type(out[0]["x"]) == str
        assert type(out[0]["y"]) == int
        assert type(out[0]["z"]) == float
        assert out[1]["x"] == "test2"
        assert out[1]["y"] == 22
        assert out[1]["z"] == 22.0
        assert type(out[1]["x"]) == str
        assert type(out[1]["y"]) == int
        assert type(out[1]["z"]) == float
        assert out[2]["x"] == "test3"
        assert out[2]["y"] == 12
        assert out[2]["z"] == 12.0
        assert type(out[2]["x"]) == str
        assert type(out[2]["y"]) == int
        assert type(out[2]["z"]) == float

    def test_get_np_keys(self):
        out = self.pipeline._get_np_keys({"x": "test", "y": np.int64(42), "z": np.float64(42.0)})
        assert out == ["y", "z"]

    def test_map_np_keys(self):
        out = self.pipeline._map_np_keys({"x": "test", "y": np.int64(42), "z": np.float64(42.0)})
        assert type(out["x"]) == str
        assert type(out["y"]) == int
        assert type(out["z"]) == float
