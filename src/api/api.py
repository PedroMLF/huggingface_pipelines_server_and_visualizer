import logging
import os
from typing import Dict, List, Union

from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.custom_types import FinalPrediction
from src.pipelines.utils import init_pipeline

app = FastAPI()

# Set logger
logger = logging.getLogger("logger")
log_formatter = logging.Formatter("[%(asctime)s] - %(name)s - %(levelname)s - %(message)s")

# Set logger - console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Set logger - logging levels
logger.setLevel(logging.DEBUG)

# Initialize config
root_path = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
config = OmegaConf.load(os.path.join(root_path, "config.yaml"))

# Initialize pipeline model
pipeline = init_pipeline(config)

# Define input types
class PredictInput(BaseModel):
    text: str


@app.post("/predict/")
async def predict(
    request: PredictInput,
) -> Dict[str, Union[str, List[FinalPrediction]]]:
    """Returns dictionary with a list of final predictions, and information
    about the type of pipeline and model.

    Args:
        request (PredictInput): Pydantic class with text string.

    Returns:
        Dict[str, Union[str, List[FinalPrediction]]]: Dictionary with keys
        "predictions", "type", and "model", with the corresponding values
        being a list of final predictions, the type of pipeline (e.g. "Text
        Classification Pipeline"), and model (e.g. "dslim/bert-base-NER").
    """
    output = pipeline(request.text)
    return {
        "predictions": output,
        "type": pipeline.pipeline_type,
        "model": pipeline.model,
    }


@app.post("/tokenize/")
async def tokenize(request: PredictInput) -> Dict[str, List[str]]:
    """Returns a dictionary with a list of full-word tokens, using the pipeline
    tokenizer.

    Args:
        request (PredictInput): Pydantic class, with text string.

    Returns:
        Dict[str, List[str]]: Dictionary with key "tokens", with the
        corresponding value being a list of tokens.
    """
    tokens = pipeline.tokenize_text(request.text)
    return {"tokens": tokens}
