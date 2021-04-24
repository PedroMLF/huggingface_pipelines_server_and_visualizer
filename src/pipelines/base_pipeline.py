from typing import Any, Dict, List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from transformers import pipeline

from src.custom_types import (
    FinalPrediction,
    RawPrediction,
)

# The task defining which pipeline will be returned. Currently accepted tasks are:
# "feature-extraction": will return a FeatureExtractionPipeline.
# "sentiment-analysis": will return a TextClassificationPipeline.
# "ner": will return a TokenClassificationPipeline.
# "question-answering": will return a QuestionAnsweringPipeline.
# "fill-mask": will return a FillMaskPipeline.
# "summarization": will return a SummarizationPipeline.
# "translation_xx_to_yy": will return a TranslationPipeline.
# "text2text-generation": will return a Text2TextGenerationPipeline.
# "text-generation": will return a TextGenerationPipeline.
# "zero-shot-classification:: will return a ZeroShotClassificationPipeline.
# "conversational": will return a ConversationalPipeline.
PIPELINE_TO_TASK_MAP = {
    "TokenClassificationPipeline": "ner",
    "TextClassificationPipeline": "sentiment-analysis",
}


class BasePipeline:
    def __init__(self, config: Union[DictConfig, ListConfig]):
        """Initializes an instance of BasePipeline. It initializes an
        HuggingFace pipeline, and an empty list of numpy keys.

        Args:
            config (Union[DictConfig, ListConfig]): OmegaConf config.
        """
        self.model = config.model
        self.hf_pipeline = config.pipeline

        # Init pipeline
        self.pipeline = pipeline(task=PIPELINE_TO_TASK_MAP[self.hf_pipeline], model=self.model)

        # Get tokenizer
        self.tokenizer = self.pipeline.tokenizer

        # Get tokenizer prefix
        self.prefix = self.pipeline.tokenizer._tokenizer.decoder.prefix

        # Define numpy keys to be mapped
        self.np_keys: List[str] = []

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text, by first getting the input_ids, converting them into
        tokens, and then joining together the subtokens.

        Args:
            text (str): User input text.

        Returns:
            List[str]: List of tokens.
        """
        input_ids = self.tokenizer(text)["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
        # TODO: Make this robust to more models, not only BERT models.
        string = (" ".join(tokens)).replace(" {}".format(self.prefix), "")
        return string.split()

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
