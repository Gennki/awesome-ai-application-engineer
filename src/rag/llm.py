"""OpenAI 兼容的 LLM 调用工具。"""

import os
from dataclasses import dataclass

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_RETRIES = 5


@dataclass(slots=True)
class LLMClient:
    """封装 OpenAI 兼容的聊天补全调用。"""

    client: OpenAI
    model: str = DEFAULT_MODEL
    temperature: float = 0

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = 0,
    ) -> "LLMClient":
        configured_api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not configured_api_key:
            raise ValueError("OPENAI_API_KEY is required")

        configured_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        configured_model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        return cls(
            client=OpenAI(
                api_key=configured_api_key,
                base_url=configured_base_url,
                max_retries=max_retries,
            ),
            model=configured_model,
            temperature=temperature,
        )

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """发送一次聊天请求并返回模型生成的文本。"""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must not be blank")

        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=system_prompt,
                )
            )
        messages.append(
            ChatCompletionUserMessageParam(role="user", content=prompt)
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        if not response.choices:
            raise ValueError("LLM returned no choices")
        return response.choices[0].message.content or ""
