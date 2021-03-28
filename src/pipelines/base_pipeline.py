from typing import Any, Dict, List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from transformers import pipeline

from src.custom_types import (
    FinalPrediction,
    RawPrediction,
)


class BasePipeline:
    def __init__(self, config: Union[DictConfig, ListConfig]):
        self.model = config.model
        self.task = config.task

        # Init pipeline
        self.pipeline = pipeline(task=self.task, model=self.model)

        # Define numpy keys to be mapped
        self.np_keys: List[str] = []

    def __call__(self, text: str):
        raise NotImplementedError

    def _map_all_np_keys(self, outputs: List[RawPrediction]) -> List[FinalPrediction]:
        if not self.np_keys:
            self.np_keys = self._get_np_keys(outputs[0])
        if self.np_keys:
            outputs = [self._map_np_keys(o) for o in outputs]
        return outputs

    def _get_np_keys(self, dictionary: RawPrediction) -> List[str]:
        keys = [k for k, v in dictionary.items() if "numpy" in str(type(v))]
        return keys

    def _map_np_keys(self, dictionary: RawPrediction) -> FinalPrediction:
        for k in self.np_keys:
            dictionary[k] = dictionary[k].item()
        return dictionary
