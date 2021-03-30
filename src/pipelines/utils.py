import logging
from typing import Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from src.pipelines.text_classification_pipeline import (
    TextClassificationPipeline,
)
from src.pipelines.token_classification_pipeline import (
    TokenClassificationPipeline,
)

logger = logging.getLogger("logger")


def init_pipeline(
    config: Union[DictConfig, ListConfig],
) -> Union[TextClassificationPipeline, TokenClassificationPipeline]:
    """Initializes an HuggingFace pipeline.

    Args:
        config (Union[DictConfig, ListConfig]): OmegaConf config.

    Raises:
        NotImplementedError: Raises error when using a non-implemented task.

    Returns:
        Union[TextClassificationPipeline, TokenClassificationPipeline]:
        Instance of "full" pipeline.
    """

    if config.task == "ner":
        logger.info("Initializing Token Classification pipeline...")
        pipeline = TokenClassificationPipeline(config)
    elif config.task == "sentiment-analysis":
        logger.info("Initializing Text Classification pipeline...")
        pipeline = TextClassificationPipeline(config)
    else:
        raise NotImplementedError

    return pipeline
