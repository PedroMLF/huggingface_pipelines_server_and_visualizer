import json
import numpy as np

from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from src.api.api import app
from src.pipelines.utils import init_pipeline

# Initialize pipeline model
MODEL_NAME = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
TASK_NAME = "ner"
config = OmegaConf.create({"task": TASK_NAME, "model": MODEL_NAME})
pipeline = init_pipeline(config)

client = TestClient(app)

# Make tests async: https://fastapi.tiangolo.com/advanced/async-tests/
def test_predict():
    response = client.post(
        "/predict/",
        json={"text": "Lisbon is a pretty city."},
    )
    assert response.status_code == 200
    response = response.json()
    assert response["type"] == "Token Classification Pipeline"
    assert response["model"] == "dslim/bert-base-NER"
    assert response["predictions"][0]["entity_group"] == "LOC"
    assert np.isclose(response["predictions"][0]["score"], 0.99, 0.1)
    assert response["predictions"][0]["word"] == "Lisbon"
    assert response["predictions"][0]["start"] == 0
    assert response["predictions"][0]["end"] == 6


def test_tokenize_simple_sentence():
    response = client.post(
        "/tokenize/",
        json={"text": "Test sentence with different parts and stuff."},
    )
    assert response.status_code == 200
    assert response.json() == {
        "tokens": ["Test", "sentence", "with", "different", "parts", "and", "stuff", "."]
    }


def test_tokenize_noisy_sentence():
    response = client.post(
        "/tokenize/",
        json={"text": "Uma frase with several #### symbols!! uaihsdausi..!!"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "tokens": [
            "Uma",
            "frase",
            "with",
            "several",
            "#",
            "#",
            "#",
            "#",
            "symbols",
            "!",
            "!",
            "uaihsdausi",
            ".",
            ".",
            "!",
            "!",
        ]
    }
