from omegaconf import OmegaConf

from src.pipelines.token_classification_pipeline import TokenClassificationPipeline

MODEL_NAME = "dslim/bert-base-NER"
PIPELINE_NAME = "TokenClassificationPipeline"


class TestTokenClassificationPipeline:
    def setup_class(cls):
        config = OmegaConf.create({"pipeline": PIPELINE_NAME, "model": MODEL_NAME})
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

    def test_call_with_diacritics(self):
        out = self.pipeline("He is Doctor Úlmán Schütze.")
        assert len(out) == 1
        assert out[0]["word"] == "Úlmán Schütze"

        out = self.pipeline("They are António Seráfim and Barack Obama!")
        assert len(out) == 2
        assert out[0]["word"] == "António Seráfim"
        assert out[1]["word"] == "Barack Obama"
