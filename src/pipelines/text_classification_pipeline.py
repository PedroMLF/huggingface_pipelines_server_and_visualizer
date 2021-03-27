from src.pipelines.base_pipeline import BasePipeline


class TextClassificationPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        self.pipeline_type = "Text Classification Pipeline"

    def __call__(self, text: str):

        output = self.pipeline(text)

        if output:
            output = self._map_all_np_keys(output)

        return output
