import json
import os

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

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

assert os.getenv("LORA_NAMES") is not None, "LORA_NAMES is not set"
assert os.getenv("LORA_URIS") is not None, "LORA_URIS is not set"

LORA_NAMES = os.getenv("LORA_NAMES").split(",")
LORA_URIS = os.getenv("LORA_URIS").split(",")


def test_add_loras():
    for lora_name, s3_uri in zip(LORA_NAMES, LORA_URIS):
        data = {
            "endpoint": "/loras",
            "payload": {
                "lora_name": lora_name,
                "s3_uri": s3_uri,
            },
        }
        response = requests.post(ENDPOINT, headers=HEADERS, json=data)

        print(response.text)
        assert response.status_code == 200
