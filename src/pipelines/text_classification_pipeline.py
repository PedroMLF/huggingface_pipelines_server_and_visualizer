from typing import List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from src.custom_types import FinalPrediction
from src.pipelines.base_pipeline import BasePipeline


class TextClassificationPipeline(BasePipeline):
    def __init__(self, config: Union[DictConfig, ListConfig]):
        super().__init__(config)
        self.pipeline_type = "Text Classification Pipeline"

    def __call__(self, text: str) -> List[FinalPrediction]:

        output = self.pipeline(text)

        if output:
            output = self._map_all_np_keys(output)

        return output
