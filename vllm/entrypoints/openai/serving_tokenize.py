import uuid
import codecs
from vllm.logger import init_logger
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (TokenizeResponseCustom,
                                              TokenizeCompletionRequest)
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRAModulePath
from typing import List, Optional
from vllm.config import ModelConfig

logger = init_logger(__name__)


class OpenAIServingTokenize(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 model_config: ModelConfig,
                 served_model_names: List[str],
                 response_role: str,
                 lora_modules: Optional[List[LoRAModulePath]] = None,
                 chat_template: Optional[str] = None):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules)
        self.response_role = response_role
        self._load_chat_template(chat_template)

    async def tokenize(self, request: TokenizeCompletionRequest):
        """
        Tokenize API that doesn't exist in OpenAI's API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        has_messages = hasattr(request, 'messages') and request.messages
        has_prompt = hasattr(request, 'prompt') and request.prompt

        if has_messages:
            if has_prompt:
                return self.create_error_response(
                    "Either `prompt` or `messages` should be provided.")

            chat_prompt = self.tokenizer.apply_chat_template(
                conversation=request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
            input_ids = self.tokenizer(chat_prompt).input_ids
        else:
            if has_messages or not has_prompt:
                return self.create_error_response(
                    "Either `prompt` or `messages` should be provided.")

            input_ids = self.tokenizer(request.prompt).input_ids

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        text = self.tokenizer.decode(input_ids)
        tokens_response = [{t: i} for t, i in zip(tokens, input_ids)]

        request_id = str(uuid.uuid4())

        logger.info(f"Tokenize request: {request_id}")

        return TokenizeResponseCustom(id=request_id,
                                      n_tokens=len(input_ids),
                                      text=text,
                                      tokens=tokens_response,
                                      model=request.model)

    def _load_chat_template(self, chat_template: Optional[str]):
        tokenizer = self.tokenizer

        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
            except OSError as e:
                JINJA_CHARS = "{}\n"
                if not any(c in chat_template for c in JINJA_CHARS):
                    msg = (f"The supplied chat template ({chat_template}) "
                           f"looks like a file path, but it failed to be "
                           f"opened. Reason: {e}")
                    raise ValueError(msg) from e

                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info("Using supplied chat template:\n%s",
                        tokenizer.chat_template)
        elif tokenizer.chat_template is not None:
            logger.info("Using default chat template:\n%s",
                        tokenizer.chat_template)
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")
