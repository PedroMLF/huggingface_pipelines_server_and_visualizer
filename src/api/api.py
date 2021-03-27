import logging
import os

from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.pipelines.utils import init_pipeline

app = FastAPI()

# Set logger
logger = logging.getLogger("logger")
log_formatter = logging.Formatter(
    "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
)

# Set logger - console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Set logger - logging levels
logger.setLevel(logging.DEBUG)

# Initialize config
root_path = os.path.dirname(os.path.abspath(__file__))
config = OmegaConf.load(os.path.join(root_path, "api_config.yaml"))

# Initialize pipeline model
pipeline = init_pipeline(config)

# Define input types
class TokenClassificationInput(BaseModel):
    text: str


@app.post("/predict/")
async def predict_item(request: TokenClassificationInput):
    output = pipeline(request.text)
    return output
