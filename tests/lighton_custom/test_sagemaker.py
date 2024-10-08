import json

import pytest
import requests
from openai import OpenAI

HOST = "http://localhost:8000/v1"
ENDPOINT = "http://localhost:8000/invocations"

openai_api_key = "EMPTY"
openai_api_base = HOST

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

model_ids = [m.id for m in models.data]

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def test_list_models():
    data = {
        "endpoint": "/v1/models",
        "payload": {},
    }
    response = requests.post(ENDPOINT, headers=HEADERS, json=data)

    assert response.status_code == 200

    result = response.json()

    print(json.dumps(result, indent=4, sort_keys=True))


@pytest.mark.parametrize(
    "model",
    model_ids,
)
def test_completion(model):
    print(f"=== Completion (model={model}) ===")
    data = {
        "endpoint": "/v1/completions",
        "payload": {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 100,
            "prompt":
            "Give me a character named Charles with a strength of 124.",
            "echo": False,
        },
    }
    response = requests.post(ENDPOINT, headers=HEADERS, json=data)

    assert response.status_code == 200

    result = response.json()

    print(json.dumps(result, indent=4, sort_keys=True))


@pytest.mark.parametrize(
    "model",
    model_ids,
)
def test_chat(model):
    print(f"=== Chat (model={model}) ===")
    data = {
        "endpoint": "/v1/chat/completions",
        "payload": {
            "model":
            model,
            "temperature":
            0.0,
            "max_tokens":
            100,
            "messages": [
                {
                    "role":
                    "user",
                    "content":
                    "Give me a character named Charles with a strength of 124.",
                },
            ],
        },
    }
    response = requests.post(ENDPOINT, headers=HEADERS, json=data)

    assert response.status_code == 200

    result = response.json()

    print(json.dumps(result, indent=4, sort_keys=True))
