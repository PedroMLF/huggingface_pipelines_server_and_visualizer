import logging

from src.pipelines.token_classification_pipeline import (
    TokenClassificationPipeline,
)

logger = logging.getLogger("logger")


def init_pipeline(config):

    if config.task == "ner":
        logger.info("Initializing Token Classification pipeline...")
        pipeline = TokenClassificationPipeline(config)
    else:
        raise NotImplementedError

    return pipeline
