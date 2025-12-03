from typing import Any, final, override

import httpx
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI
from pydantic import Field

from bs_translator_backend.models.app_config import AppConfig
from bs_translator_backend.utils.logger import get_logger

logger = get_logger("QwenVllm")


@final
class QwenVllm(CustomLLM):
    """
    Custom LLM implementation for Qwen using OpenAI-compatible API.

    Attributes:
        client: OpenAI client instance.
        config: Application configuration.
        last_log: Last log message as a string.
    """

    client: OpenAI
    http_client: httpx.AsyncClient
    config: AppConfig
    last_log: str = Field(default="", description="Last log message")

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize QwenVllm with the given configuration.

        Args:
            config: Application configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base_url,
        )
        http_client = httpx.AsyncClient(timeout=5.0)
        # Call the superclass __init__ first to ensure all base attributes are initialized
        super().__init__(config=config, client=client)

    @property
    @override
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.config.llm_model, is_chat_model=True, is_function_calling_model=False
        )

    @override
    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Complete a prompt with optional tool usage.

        Args:
            prompt: The input prompt
            tools: List of tools available to the model
            tool_choice: Controls how the model uses tools. Can be "none", "auto", or a specific tool name
            **kwargs: Additional parameters to pass to the completion API

        Returns:
            CompletionResponse with text and raw API response
        """
        completion = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt + " /no_think"}],
            presence_penalty=1.5,
            top_p=0.8,
            temperature=0.7,
            extra_body={"top_k": 20},
            max_tokens=8_000,
        )

        choice = completion.choices[0]

        if choice.finish_reason == "length":
            logger.warning("Completion stopped due to length limit.")

        message = completion.choices[0].message
        output: str = message.content or ""  # Handle None case explicitly

        try:
            self.last_log = completion.choices[0].model_dump_json()
        except Exception:
            logger.exception("Error in model_dump_json")
            self.last_log = str(completion.choices[0])

        return CompletionResponse(text=output, raw=completion)

    @override
    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """
        Stream complete a prompt with optional tool usage.

        Args:
            prompt: The input prompt
            tools: List of tools available to the model
            tool_choice: Controls how the model uses tools
            **kwargs: Additional parameters to pass to the completion API

        Yields:
            CompletionResponse chunks with progressive text and deltas
        """
        response = ""

        stream = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt + " /no_think"}],
            presence_penalty=1.5,
            top_p=0.8,
            temperature=0.7,
            extra_body={"top_k": 20},
            stream=True,
            max_tokens=8_000,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content.replace("ß", "ss")
                response += content
                yield CompletionResponse(text=response, delta=content)

            # Handle tool calls in streaming mode
            if (
                chunk.choices
                and hasattr(chunk.choices[0].delta, "tool_calls")
                and chunk.choices[0].delta.tool_calls
            ):
                # For tool calls in streaming, we just log them but actual tool execution
                # should be handled by the caller after the stream is complete
                self.last_log = f"Tool call received in chunk: {chunk.model_dump_json()}"

    async def is_ready(self) -> bool:
        response = await self.http_client.get(f"{self.config.openai_api_base_url}/health")
        if response.status_code != 200:
            raise Exception(f"LLM is not ready: {response.status_code}")
        return True
