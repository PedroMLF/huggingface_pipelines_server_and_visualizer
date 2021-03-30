from typing import List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from src.custom_types import FinalPrediction
from src.pipelines.base_pipeline import BasePipeline


class TokenClassificationPipeline(BasePipeline):
    def __init__(self, config: Union[DictConfig, ListConfig]):
        """Initializes an instance of TokenClassificationPipeline.

        Args:
            config (Union[DictConfig, ListConfig]): OmegaConf config.
        """
        super().__init__(config)
        self.pipeline_type = "Token Classification Pipeline"

    def __call__(self, text: str) -> List[FinalPrediction]:
        """Returns list of dictionaries, each corresponding to a final
        prediction. It runs HuggingFace's corresponding pipeline, and maps
        numpy types to regular types, when necessary. It also groups
        consecutive entities of the same type as a single entity.

        Args:
            text (str): Input text string.

        Returns:
            List[FinalPrediction]: List of dictionaries, each corresponding to
            a final prediction.
        """

        output = self.pipeline(text)
        output = self.pipeline.group_entities(output)

        if output:
            output = self._map_all_np_keys(output)

        return output
