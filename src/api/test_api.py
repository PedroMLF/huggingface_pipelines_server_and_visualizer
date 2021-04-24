import json
import numpy as np

from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from src.api.api import app
from src.pipelines.utils import init_pipeline

# Initialize FastAPI TestClient
client = TestClient(app)

# Make tests async: https://fastapi.tiangolo.com/advanced/async-tests/
def test_predict():
    response = client.post(
        "/predict/",
        json={"text": "Lisbon is a pretty city."},
    )
    assert response.status_code == 200
    response = response.json()
    assert type(response["type"]) == str
    assert type(response["model"]) == str
    assert type(response["predictions"]) == list


def test_tokenize_sentence():
    response = client.post(
        "/tokenize/",
        json={"text": "Test sentence and stuff."},
    )
    assert response.status_code == 200
    assert response.json() == {"tokens": ["Test", "sentence", "and", "stuff", "."]}
