import json
import uuid

import openai
import pytest
import requests
from openai import OpenAI

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

schema = """{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "maxLength": 10,
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}"""
regex = "(-)?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-][0-9]+)?"

choices = ["It is Joe Biden!", "It is Emmanual Macron!", "It is Angela Merkel!"]


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


@pytest.mark.parametrize(
    "model,stream",
    [(model, stream) for model in model_ids for stream in [False, True]],
)
def test_completion(model, stream):
    print(f"=== Completion (model={model}, stream={stream}) ===")
    completion = client.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=100,
        prompt="Give me a character named Charles with a strength of 124.",
        echo=False,
        stream=stream,
    )
    if stream:
        for c in completion:
            print(c.model_dump_json())
    else:
        print(completion.model_dump_json())


@pytest.mark.parametrize(
    "model,stream",
    [(model, stream) for model in model_ids for stream in [False, True]],
)
def test_completion_json(model, stream):
    print(f"=== Completion JSON (model={model}, stream={stream}) ===")
    completion = client.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=100,
        prompt="Give me a character named Charles with a strength of 124.",
        echo=False,
        stream=stream,
        extra_body={
            "json_schema": schema,
        },
    )
    if stream:
        for c in completion:
            print(c.model_dump_json())
    else:
        print(completion.model_dump_json())
        print(json.loads(completion.choices[0].text))


@pytest.mark.parametrize(
    "model,stream",
    [(model, stream) for model in model_ids for stream in [False, True]],
)
def test_completion_regex(model, stream):
    print(f"=== Completion Regex (model={model}, stream={stream}) ===")
    completion = client.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=100,
        prompt="Give me the 10 first digits of PI: ",
        echo=False,
        stream=stream,
        extra_body={
            "regex": regex,
        },
    )
    if stream:
        for c in completion:
            print(c.model_dump_json())
    else:
        print(completion.model_dump_json())


@pytest.mark.parametrize(
    "model,stream",
    [(model, stream) for model in model_ids for stream in [False, True]],
)
def test_completion_guided_choice(model, stream):
    print(f"=== Completion guided choice (model={model}, stream={stream}) ===")
    completion = client.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=100,
        prompt="Who is the President of the United States?",
        echo=False,
        stream=stream,
        extra_body={
            "guided_choice": choices,
        },
    )
    if stream:
        for c in completion:
            print(c.model_dump_json())
    else:
        print(completion.model_dump_json())
        assert completion.choices[0].text in choices


@pytest.mark.parametrize(
    "model,stream",
    [(model, stream) for model in model_ids for stream in [False, True]],
)
def test_chat(model, stream):
    print(f"=== Chat (model={model}, stream={stream}) ===")
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Give me a character named Charles with a strength of 124.",
            },
        ],
        model=model,
        stream=stream,
        max_tokens=100,
        temperature=0.0,
        stop=["</s>"],
    )
    if stream:
        for c in completion:
            print(c.model_dump_json())
    else:
        print(completion.model_dump_json())


@pytest.mark.parametrize(
    "model,stream",
    [(model, stream) for model in model_ids for stream in [False, True]],
)
def test_chat_json(model, stream):
    print(f"=== Chat JSON (model={model}, stream={stream}) ===")
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Give me a character named Charles with a strength of 124.",
            },
        ],
        model=model,
        stream=stream,
        max_tokens=100,
        temperature=0.0,
        stop=["</s>"],
        extra_body={
            "json_schema": schema,
        },
    )
    if stream:
        for c in completion:
            print(c.model_dump_json())
    else:
        print(completion.model_dump_json())
        print(json.loads(completion.choices[0].message.content))


@pytest.mark.parametrize(
    "model,stream",
    [(model, stream) for model in model_ids for stream in [False, True]],
)
def test_chat_regex(model, stream):
    print(f"=== Chat Regex (model={model}, stream={stream}) ===")
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Give me the 10 first digits of PI."},
        ],
        model=model,
        stream=stream,
        max_tokens=100,
        temperature=0.0,
        stop=["</s>"],
        extra_body={
            "regex": regex,
        },
    )
    if stream:
        for c in completion:
            print(c.model_dump_json())
    else:
        print(completion.model_dump_json())


@pytest.mark.parametrize(
    "model,stream",
    [(model, stream) for model in model_ids for stream in [False, True]],
)
def test_chat_guided_choice(model, stream):
    print(f"=== Chat guided choice (model={model}, stream={stream}) ===")
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Who is the President of the United States?",
            },
        ],
        model=model,
        stream=stream,
        max_tokens=100,
        temperature=0.0,
        stop=["</s>"],
        extra_body={
            "guided_choice": choices,
        },
    )
    if stream:
        for c in completion:
            print(c.model_dump_json())
    else:
        print(completion.model_dump_json())
        assert completion.choices[0].message.content in choices


def test_json_schema_and_guided_regex():
    with pytest.raises(openai.BadRequestError):
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Give me a character named Charles with a strength of 124.",
                },
            ],
            model=model,
            stream=False,
            max_tokens=100,
            temperature=0.0,
            stop=["</s>"],
            extra_body={
                "json_schema": schema,
                "guided_regex": regex,
            },
        )


@pytest.mark.parametrize("model", model_ids)
def test_tokenize(model):
    url = f"{HOST}/tokenize"
    data = {
        "prompt": "What is the weather today?",
        "model": model,
    }

    response = requests.post(url, headers=HEADERS, json=data)

    assert response.status_code == 200

    result = response.json()
    assert is_uuid4(result["id"])
    assert result["model"] == model

    print(json.dumps(result, indent=4, sort_keys=True))


@pytest.mark.parametrize("model", model_ids)
def test_chat_tokenize(model):
    url = f"{HOST}/tokenize"
    data = {
        "messages": [{"role": "user", "content": "What is the weather today?"}],
        "model": model,
    }

    response = requests.post(url, headers=HEADERS, json=data)

    assert response.status_code == 200

    result = response.json()
    assert is_uuid4(result["id"])
    assert result["model"] == model

    print(json.dumps(result, indent=4, sort_keys=True))


@pytest.mark.parametrize("model", model_ids)
def test_invalid_payload_tokenize(model):
    url = f"{HOST}/tokenize"
    data = {
        "model": model,
    }

    response = requests.post(url, headers=HEADERS, json=data)

    assert response.status_code == 400

    result = response.json()

    print(json.dumps(result, indent=4, sort_keys=True))
