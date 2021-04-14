from omegaconf import OmegaConf

from src.pipelines.text_classification_pipeline import TextClassificationPipeline

MODEL_NAME = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
TASK_NAME = "sentiment-analysis"


class TestTextClassificationPipeline:
    def setup_class(cls):
        config = OmegaConf.create({"task": TASK_NAME, "model": MODEL_NAME})
        cls.pipeline = TextClassificationPipeline(config)

    def teardown_class(cls):
        pass

    def test_correct_pipeline_type(self):
        assert self.pipeline.pipeline_type == "Text Classification Pipeline"

    def test_call(self):
        out = self.pipeline("Lisbon is a great and amazing city!")
        assert len(out) == 1
        assert list(out[0].keys()) == ["label", "score"]
        assert out[0]["label"] in ["POSITIVE", "NEGATIVE"]
        assert type(out[0]["score"]) == float
