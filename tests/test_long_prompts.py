import os
import uuid

import pytest
import torch
from openai import OpenAI
from transformers import AutoTokenizer

HOST = "http://localhost:8000"

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


def is_uuid4(string: str) -> bool:
    try:
        uuid.UUID(string, version=4)
        return True
    except ValueError:
        return False


BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "gpt2")
PROMPT_LEN = int(os.getenv("PROMPT_LEN", 1024))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
vocab_size = tokenizer.vocab_size
ids = torch.randint(0, vocab_size, (1, PROMPT_LEN))


@pytest.mark.parametrize("model", model_ids)
def test_completion(model):
    print(f"=== Completion (model={model}) ===")
    completion = client.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=100,
        prompt=ids.tolist()[0],
        echo=False,
        stream=False,
        extra_body={
            "ignore_eos": True,
        },
    )
    print(completion.model_dump_json())
