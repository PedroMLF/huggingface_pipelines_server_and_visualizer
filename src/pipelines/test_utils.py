import pytest
from omegaconf import OmegaConf

from src.pipelines.utils import init_pipeline


class TestUtils:
    def setup_class(cls):
        pass

    def teardown_class(cls):
        pass

    def test_init_test_token_classification_pipeline(self):
        config = OmegaConf.create(
            {
                "task": "ner",
                "model": "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english",
            }
        )
        pipeline = init_pipeline(config)
        assert pipeline.pipeline_type == "Token Classification Pipeline"

    def test_init_test_text_classification_pipeline(self):
        config = OmegaConf.create(
            {
                "task": "sentiment-analysis",
                "model": "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
            }
        )
        pipeline = init_pipeline(config)
        assert pipeline.pipeline_type == "Text Classification Pipeline"

    def test_invalid_init_raises_not_implemented_error(self):
        config = OmegaConf.create({"task": "x", "model": "y"})
        with pytest.raises(NotImplementedError):
            pipeline = init_pipeline(config)
