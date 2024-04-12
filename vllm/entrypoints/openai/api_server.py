import asyncio
import importlib
import inspect
import os
from contextlib import asynccontextmanager
from http import HTTPStatus

import fastapi
import uvicorn
import boto3
import tarfile
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.usage.usage_lib import UsageContext
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    InvocationRequest,
    TokenizeCompletionRequest,
    AddLoRARequest,
)
from vllm.logger import init_logger
from vllm.entrypoints.openai.serving_engine import LoRA
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_tokenize import OpenAIServingTokenize

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
openai_serving_tokenize: OpenAIServingTokenize = None
logger = init_logger(__name__)

s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-west-2"))
LORA_FOLDER_PATH = "/tmp/lora_modules"


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/ping")
async def ping() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/invocations")
async def invocations(request: InvocationRequest, raw_request: Request):
    if request.endpoint == "/models":
        return await show_available_models()
    elif request.endpoint == "/chat/completions":
        return await create_chat_completion(request.payload, raw_request)
    elif request.endpoint == "/completions":
        return await create_completion(request.payload, raw_request)
    elif request.endpoint == "/tokenize":
        return await tokenize(request.payload)
    elif request.endpoint == "/loras":
        return await add_lora(request.payload, raw_request)
    else:
        err = openai_serving_chat.create_error_response(message=f"Endpoint {request.endpoint} not found")
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.NOT_FOUND)


@app.get("/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.post("/loras")
async def add_lora(request: AddLoRARequest, raw_request: Request):
    for lora_request in openai_serving_chat.lora_requests:
        if lora_request.lora_name == request.lora_name:
            return {"success": True}
            
    lora_dir = f"{LORA_FOLDER_PATH}/{request.lora_name}"

    # if lora path do not exists create it
    if not os.path.exists(lora_dir):
        os.makedirs(lora_dir)

    # download lora module from s3
    s3_client.download_file(request.s3_bucket, request.s3_key, f"{lora_dir}/model.tar.gz")

    # extract lora module
    with tarfile.open(f"{lora_dir}/model.tar.gz", "r:gz") as tar:
        tar.extractall(path=lora_dir)

    # remove tar file
    os.remove(f"{lora_dir}/model.tar.gz")

    lora = LoRA(request.lora_name, lora_dir)
    await openai_serving_completion._add_lora(lora=lora)
    await openai_serving_chat._add_lora(lora=lora)
    return {"success": True}


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@app.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/tokenize")
async def tokenize(request: TokenizeCompletionRequest):
    generator = await openai_serving_tokenize.tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if not request.url.path.startswith(f"{root_path}/"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info(f"vLLM API server version {vllm.__version__}")
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    openai_serving_chat = OpenAIServingChat(engine, served_model,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model, args.lora_modules)
    openai_serving_tokenize = OpenAIServingTokenize(engine, served_model,
                                            args.response_role,
                                            args.chat_template)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
