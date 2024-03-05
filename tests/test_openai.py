import openai
from openai import OpenAI
import json
import pytest
import requests
import uuid
import os

HOST = "http://localhost:8000"

openai_api_key = "EMPTY"
openai_api_base = HOST

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

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


@pytest.mark.parametrize("stream", [False, True])
def test_completion(stream):
    print(f"=== Completion (stream={stream}) ===")
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


@pytest.mark.parametrize("stream", [False, True])
def test_completion_json(stream):
    print(f"=== Completion JSON (stream={stream}) ===")
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


@pytest.mark.parametrize("stream", [False, True])
def test_completion_regex(stream):
    print(f"=== Completion Regex (stream={stream}) ===")
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


@pytest.mark.parametrize("stream", [False, True])
def test_completion_guided_choice(stream):
    print(f"=== Completion guided choice (stream={stream}) ===")
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


@pytest.mark.parametrize("stream", [False, True])
def test_chat(stream):
    print(f"=== Chat (stream={stream}) ===")
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


@pytest.mark.parametrize("stream", [False, True])
def test_chat_json(stream):
    print(f"=== Chat JSON (stream={stream}) ===")
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


@pytest.mark.parametrize("stream", [False, True])
def test_chat_regex(stream):
    print(f"=== Chat Regex (stream={stream}) ===")
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


@pytest.mark.parametrize("stream", [False, True])
def test_chat_guided_choice(stream):
    print(f"=== Chat guided choice (stream={stream}) ===")
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


@pytest.mark.parametrize("model", [model])
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


@pytest.mark.parametrize("model", [model])
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


@pytest.mark.parametrize("model", [model])
def test_invalid_payload_tokenize(model):
    url = f"{HOST}/tokenize"
    data = {
        "model": model,
    }

    response = requests.post(url, headers=HEADERS, json=data)

    assert response.status_code == 400

    result = response.json()

    print(json.dumps(result, indent=4, sort_keys=True))
