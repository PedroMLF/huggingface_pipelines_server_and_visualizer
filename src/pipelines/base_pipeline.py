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
        """Initializes an instance of BasePipeline. It initializes an
        HuggingFace pipeline, and an empty list of numpy keys.

        Args:
            config (Union[DictConfig, ListConfig]): OmegaConf config.
        """
        self.model = config.model
        self.task = config.task

        # Init pipeline
        self.pipeline = pipeline(task=self.task, model=self.model)

        # Define numpy keys to be mapped
        self.np_keys: List[str] = []

    def __call__(self, text: str):
        """__call__ method to be implemented by subclasses.

        Args:
            text (str): Input text string.

        Raises:
            NotImplementedError: Method to be implemented by subclasses.
        """
        raise NotImplementedError

    def _map_all_np_keys(self, outputs: List[RawPrediction]) -> List[FinalPrediction]:
        """Returns list of dictionaries, each corresponding to a final
        prediction, after mapping numpy types to regular types, when necessary.

        Args:
            outputs (List[RawPrediction]): List of dictionaries, each
            corresponding to a raw prediction.

        Returns:
            List[FinalPrediction]: List of dictionaries, each corresponding to
            a final prediction.
        """
        if not self.np_keys:
            self.np_keys = self._get_np_keys(outputs[0])
        if self.np_keys:
            outputs = [self._map_np_keys(o) for o in outputs]
        return outputs

    def _get_np_keys(self, dictionary: RawPrediction) -> List[str]:
        """Returns keys of a dictionary that correspond to numpy types.

        Args:
            dictionary (RawPrediction): Dictionary with data corresponding to
            a raw prediction.

        Returns:
            List[str]: List with string keys corresponding to numpy types.
        """
        keys = [k for k, v in dictionary.items() if "numpy" in str(type(v))]
        return keys

    def _map_np_keys(self, dictionary: RawPrediction) -> FinalPrediction:
        """Casts values of a dictionary from numpy types to the corresponding
        regular python types, using numpy's .item() on those values.

        Args:
            dictionary (RawPrediction): Dictionary with data corresponding to
            a raw prediction.

        Returns:
            FinalPrediction: Dictionary with data corresponding to a final
            prediction.
        """
        for k in self.np_keys:
            dictionary[k] = dictionary[k].item()
        return dictionary
