from typing import List
import uuid
from vllm.logger import init_logger
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    TokenizeResponse,
    TokenizeCompletionRequest
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing

logger = init_logger(__name__)


class OpenAIServingTokenize(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model_names: List[str],
                 response_role: str,
                 chat_template=None):
        super().__init__(engine=engine, served_model_names=served_model_names, lora_modules=None)
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
        tokens_response = [{t:i} for t,i in zip(tokens, input_ids)]

        request_id = str(uuid.uuid4())

        logger.info(f"Tokenize request: {request_id}")

        return TokenizeResponse(id=request_id, n_tokens=len(input_ids), text=text, tokens=tokens_response, model=request.model)

    def _load_chat_template(self, chat_template):
        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    self.tokenizer.chat_template = f.read()
            except OSError:
                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                self.tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info(
                f"Using supplied chat template:\n{self.tokenizer.chat_template}"
            )
        elif self.tokenizer.chat_template is not None:
            logger.info(
                f"Using default chat template:\n{self.tokenizer.chat_template}"
            )
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")
