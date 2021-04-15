from omegaconf import OmegaConf

from src.pipelines.token_classification_pipeline import TokenClassificationPipeline

MODEL_NAME = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
TASK_NAME = "ner"


class TestTokenClassificationPipeline:
    def setup_class(cls):
        config = OmegaConf.create({"task": TASK_NAME, "model": MODEL_NAME})
        cls.pipeline = TokenClassificationPipeline(config)

    def teardown_class(cls):
        pass

    def test_correct_pipeline_type(self):
        assert self.pipeline.pipeline_type == "Token Classification Pipeline"

    def test_call(self):
        out = self.pipeline("Lisbon is a great city!")
        assert len(out) > 0
        assert list(out[0].keys()) == ["entity_group", "score", "word", "start", "end"]
        assert type(out[0]["score"]) == float
        assert type(out[0]["start"]) == int
        assert type(out[0]["end"]) == int