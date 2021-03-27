import logging

from src.pipelines.text_classification_pipeline import (
    TextClassificationPipeline,
)
from src.pipelines.token_classification_pipeline import (
    TokenClassificationPipeline,
)

logger = logging.getLogger("logger")


def init_pipeline(config):

    if config.task == "ner":
        logger.info("Initializing Token Classification pipeline...")
        pipeline = TokenClassificationPipeline(config)
    elif config.task == "sentiment-analysis":
        logger.info("Initializing Text Classification pipeline...")
        pipeline = TextClassificationPipeline(config)
    else:
        raise NotImplementedError

    return pipeline
