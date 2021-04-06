from typing import List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from src.custom_types import FinalPrediction
from src.pipelines.base_pipeline import BasePipeline


class TextClassificationPipeline(BasePipeline):
    def __init__(self, config: Union[DictConfig, ListConfig]):
        """Initializes an instance of TextClassificationPipeline.

        Args:
            config (Union[DictConfig, ListConfig]): OmegaConf config.
        """
        super().__init__(config)
        self.pipeline_type = "Text Classification Pipeline"

    def __call__(self, text: str) -> List[FinalPrediction]:
        """Returns list of dictionaries, each corresponding to a final
        prediction. It runs HuggingFace's corresponding pipeline, and maps
        numpy types to regular types, when necessary.

        Args:
            text (str): Input text string.

        Returns:
            List[FinalPrediction]: List of dictionaries, each corresponding
            to a final prediction.
        """

        output = self.pipeline(text)

        if output:
            output = self._map_all_np_keys(output)

        return output
