import asyncio
import importlib
import inspect
import re
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional, Set

import fastapi
import uvicorn
from fastapi import APIRouter, Request
import boto3
import tarfile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount
from pydantic import ValidationError

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (
    AddLoRARequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    DetokenizeRequest,
    DetokenizeResponse,
    EmbeddingRequest,
    ErrorResponse,
    InvocationRequest,
    TokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds

engine: AsyncLLMEngine
engine_args: AsyncEngineArgs
openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding
openai_serving_tokenization: OpenAIServingTokenization

logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()

s3_client = boto3.client("s3",
                         region_name=os.getenv("AWS_REGION", "us-west-2"))
LORA_FOLDER_PATH = "/tmp/lora_modules"


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


router = APIRouter()


def mount_metrics(app: fastapi.FastAPI):
    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app())
    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile('^/metrics(?P<path>.*)$')
    app.routes.append(metrics_route)


def parse_tokenize_request(payload: dict) -> TokenizeRequest:
    try:
        return TokenizeCompletionRequest.model_validate(payload)
    except ValidationError:
        return TokenizeChatRequest.model_validate(payload)


@router.get("/ping")
async def ping() -> Response:
    return await health()


@router.post("/invocations")
async def invocations(request: InvocationRequest, raw_request: Request):
    if request.endpoint == "/models":
        return await show_available_models()
    elif request.endpoint == "/chat/completions":
        payload = ChatCompletionRequest.model_validate(request.payload)
        return await create_chat_completion(payload, raw_request)
    elif request.endpoint == "/completions":
        payload = CompletionRequest.model_validate(request.payload)
        return await create_completion(payload, raw_request)
    elif request.endpoint == "/tokenize":
        payload = parse_tokenize_request(request.payload)
        return await tokenize(payload)
    elif request.endpoint == "/detokenize":
        payload = DetokenizeRequest.model_validate(request.payload)
        return await detokenize(payload)
    elif request.endpoint == "/embeddings":
        payload = EmbeddingRequest.model_validate(request.payload)
        return await create_embedding(payload, raw_request)
    elif request.endpoint == "/loras":
        payload = AddLoRARequest.model_validate(request.payload)
        return await add_lora(payload, raw_request)
    else:
        err = openai_serving_chat.create_error_response(
            message=f"Endpoint {request.endpoint} not found")
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.NOT_FOUND)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    generator = await openai_serving_tokenization.create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, TokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@router.post("/detokenize")
async def detokenize(request: DetokenizeRequest):
    generator = await openai_serving_tokenization.create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, DetokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@router.get("/models")
async def show_available_models():
    models = await openai_serving_completion.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.post("/loras")
async def add_lora(request: AddLoRARequest, raw_request: Request):
    for lora_request in openai_serving_chat.lora_requests:
        if lora_request.lora_name == request.lora_name:
            return JSONResponse(content={
                "error":
                f"LoRA module {request.lora_name} already exists."
            },
                                status_code=HTTPStatus.BAD_REQUEST)

    if request.s3_uri and request.local_path:
        return JSONResponse(content={
            "error":
            "Both s3_uri and local_path cannot be provided."
        },
                            status_code=HTTPStatus.BAD_REQUEST)

    if request.s3_uri is None and request.local_path is None:
        return JSONResponse(
            content={"error": "Either s3_uri or local_path must be provided."},
            status_code=HTTPStatus.BAD_REQUEST)

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
            logger.info(
                f"Downloading lora module from s3: {s3_uri} ({s3_bucket}/{s3_key})"
            )
            s3_client.download_file(s3_bucket, s3_key,
                                    f"{lora_dir}/model.tar.gz")
        except Exception as e:
            return JSONResponse(content={
                "error":
                f"Error downloading lora module from s3: {e}"
            },
                                status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

        # extract lora module
        with tarfile.open(f"{lora_dir}/model.tar.gz", "r:gz") as tar:
            tar.extractall(path=lora_dir)

        # remove tar file
        os.remove(f"{lora_dir}/model.tar.gz")

        # list directory and print files
        logger.info(f"Extracted lora module files:")
        for file in os.listdir(lora_dir):
            logger.info(f"  - {file}")

    lora = LoRAModulePath(request.lora_name, lora_dir)
    await openai_serving_completion._add_lora(lora=lora)
    await openai_serving_chat._add_lora(lora=lora)
    await openai_serving_tokenization._add_lora(lora=lora)
    return {"success": True}


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/chat/completions")
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
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@router.post("/completions")
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


@router.post("/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await openai_serving_embedding.create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        return JSONResponse(content=generator.model_dump())


def build_app(args):
    app = fastapi.FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = openai_serving_chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
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

    return app


def run_server(args, llm_engine=None):
    app = build_app(args)

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    global engine, engine_args

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (llm_engine
              if llm_engine is not None else AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.OPENAI_API_SERVER))

    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    global openai_serving_chat
    global openai_serving_completion
    global openai_serving_embedding
    global openai_serving_tokenization

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
    )
    openai_serving_embedding = OpenAIServingEmbedding(
        engine,
        model_config,
        served_model_names,
        request_logger=request_logger,
    )
    openai_serving_tokenization = OpenAIServingTokenization(
        engine,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )
    app.root_path = args.root_path

    logger.info("Available routes are:")
    for route in app.routes:
        if not hasattr(route, 'methods'):
            continue
        methods = ', '.join(route.methods)
        logger.info("Route: %s, Methods: %s", route.path, methods)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    run_server(args)
