from typing import List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from src.custom_types import FinalPrediction
from src.custom_types import RawPrediction
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

        # Fix NER entities
        self._fix_ner_entities(output)

        # Group words of the same entity
        output = self.pipeline.group_entities(output)

        if output:
            output = self._map_all_np_keys(output)

        return output

    def _fix_ner_entities(self, predictions: List[RawPrediction]) -> None:
        """Converts entities with "B-" into "I-" for words with "##". This
        happens for words with diacritics, for instance António. In that case
        "##t" would be tagged as "B-PER" instead of "I-PER", and would result
        in wrong grouped entities when calling self.pipeline.group_entities.
        We modify entities in-place, since python uses "Call by Sharing".

        Args:
            predictions (List[RawPrediction]): List of dictionaries, each
            corresponding to a raw prediction.
        """
        # TODO: Add more checks about start position of B- being the same as
        # the end position of the last entity, and having the same entity type
        for d in predictions:
            if self.prefix in d["word"] and "B-" in d["entity"]:
                d["entity"] = d["entity"].replace("B-", "I-")
