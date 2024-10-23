import os
import tarfile
from http import HTTPStatus
from typing import Any, Optional

import boto3
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import vllm.envs as envs
from vllm.entrypoints.openai.api_server import (
    chat,
    create_chat_completion,
    create_completion,
    health,
    logger,
    show_available_models,
    show_version,
    tokenize,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    LoadLoraAdapterRequest,
    TokenizeRequest,
)

s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-west-2"))
LORA_FOLDER_PATH = "/tmp/lora_modules"


class AddLoRARequest(BaseModel):
    lora_name: str
    s3_uri: Optional[str] = None
    local_path: Optional[str] = None


class InvocationRequest(BaseModel):
    endpoint: str
    payload: Optional[Any]


router = APIRouter()


@router.get("/ping")
async def ping(raw_request: Request) -> Response:
    return await health(raw_request)


@router.post("/invocations")
async def invocations(request: InvocationRequest, raw_request: Request):
    if request.endpoint == "/version":
        return await show_version()
    elif request.endpoint == "/v1/models":
        return await show_available_models(raw_request)
    elif request.endpoint == "/v1/chat/completions":
        payload = ChatCompletionRequest.model_validate(request.payload)
        return await create_chat_completion(payload, raw_request)
    elif request.endpoint == "/v1/completions":
        payload = CompletionRequest.model_validate(request.payload)
        return await create_completion(payload, raw_request)
    elif request.endpoint == "/tokenize":
        payload = TokenizeRequest.model_validate(request.payload)
        return await tokenize(payload, raw_request)
    elif request.endpoint == "/loras":
        if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
            payload = AddLoRARequest.model_validate(request.payload)
            return await add_lora(payload, raw_request)
        else:
            err = chat(raw_request).create_error_response(message="Runtime LoRA updating is disabled")
            return JSONResponse(err.model_dump(), status_code=HTTPStatus.FORBIDDEN)
    else:
        err = chat(raw_request).create_error_response(message=f"Endpoint {request.endpoint} not found")
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.NOT_FOUND)


if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
    from vllm.entrypoints.openai.api_server import load_lora_adapter

    @router.post("/loras")
    async def add_lora(request: AddLoRARequest, raw_request: Request):
        if request.s3_uri and request.local_path:
            return JSONResponse(
                content={"error": "Both s3_uri and local_path cannot be provided."}, status_code=HTTPStatus.BAD_REQUEST
            )

        if request.s3_uri is None and request.local_path is None:
            return JSONResponse(
                content={"error": "Either s3_uri or local_path must be provided."}, status_code=HTTPStatus.BAD_REQUEST
            )

        if request.local_path:
            lora_dir = request.local_path
            logger.info(f"Loading LoRA module from local path: {lora_dir}")
        else:
            lora_dir = f"{LORA_FOLDER_PATH}/{request.lora_name}"
            logger.info(f"Loading LoRA module from s3: {request.s3_uri}")
            logger.info(f"LoRA module will be stored in: {lora_dir}")

            # if lora path do not exists create it
            if not os.path.exists(lora_dir):
                logger.info(f"Creating lora module directory: {lora_dir}")
                os.makedirs(lora_dir)

            s3_uri = request.s3_uri
            s3_bucket = s3_uri.split("/")[2]
            s3_key = s3_uri.split("/", 3)[3]

            # download lora module from s3
            try:
                logger.info(f"Downloading lora module from s3: {s3_uri} ({s3_bucket}/{s3_key})")
                s3_client.download_file(s3_bucket, s3_key, f"{lora_dir}/model.tar.gz")
            except Exception as e:
                return JSONResponse(
                    content={"error": f"Error downloading lora module from s3: {e}"},
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            # extract lora module
            with tarfile.open(f"{lora_dir}/model.tar.gz", "r:gz") as tar:
                tar.extractall(path=lora_dir)

            # remove tar file
            os.remove(f"{lora_dir}/model.tar.gz")

            # list directory and print files
            logger.info("Extracted lora module files:")
            for file in os.listdir(lora_dir):
                logger.info(f"  - {file}")

        lora_adapter_request = LoadLoraAdapterRequest(lora_name=request.lora_name, lora_path=lora_dir)
        return await load_lora_adapter(lora_adapter_request, raw_request)
