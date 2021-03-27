from src.pipelines.base_pipeline import BasePipeline


class TokenClassificationPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, text: str):

        output = self.pipeline(text)
        output = self.pipeline.group_entities(output)

        if output:
            output = self._map_all_np_keys(output)

        return output
