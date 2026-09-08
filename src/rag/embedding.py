"""统一的文本向量化工具。"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ollama import Client as OllamaClient
from openai import OpenAI


DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LOCAL_MODEL = "qwen3-embedding:0.6b"
DEFAULT_ONLINE_BASE_URL = (
    "https://ws-uvye6qi8sohbokp2.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1"
)
DEFAULT_ONLINE_MODEL = "qwen3.7-text-embedding"

EmbeddingVector = list[float]


class Embedder(Protocol):
    """向量化工具需要实现的公共接口。"""

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """批量生成文本向量，并保持输入顺序。"""


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, str) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of strings")

    normalized = list(texts)
    if any(not isinstance(text, str) for text in normalized):
        raise TypeError("texts must contain only strings")
    if any(not text.strip() for text in normalized):
        raise ValueError("texts must not contain blank strings")
    return normalized


def _validate_vector_count(
    vectors: list[EmbeddingVector],
    expected_count: int,
) -> list[EmbeddingVector]:
    if len(vectors) != expected_count:
        raise ValueError(
            "embedding response count does not match input count: "
            f"expected {expected_count}, got {len(vectors)}"
        )
    return vectors


@dataclass(slots=True)
class LocalEmbedder:
    """使用本地 Ollama 服务生成文本向量。"""

    client: OllamaClient
    model: str = DEFAULT_LOCAL_MODEL

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str | None = None,
    ) -> "LocalEmbedder":
        configured_base_url = base_url or os.getenv(
            "EMBEDDING_BASE_URL",
            DEFAULT_LOCAL_BASE_URL,
        )
        configured_model = model or os.getenv(
            "EMBEDDING_MODEL",
            DEFAULT_LOCAL_MODEL,
        )
        return cls(
            client=OllamaClient(host=configured_base_url),
            model=configured_model,
        )

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        normalized = _validate_texts(texts)
        if not normalized:
            return []

        response = self.client.embed(model=self.model, input=normalized)
        vectors = [list(vector) for vector in response.embeddings]
        return _validate_vector_count(vectors, len(normalized))


@dataclass(slots=True)
class OnlineEmbedder:
    """使用阿里云百炼 OpenAI 兼容接口生成文本向量。"""

    client: OpenAI
    model: str = DEFAULT_ONLINE_MODEL

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> "OnlineEmbedder":
        configured_api_key = (
            api_key or os.getenv("EMBEDDING_API_KEY", "")
        ).strip()
        if not configured_api_key:
            raise ValueError("EMBEDDING_API_KEY is required")

        configured_base_url = base_url or os.getenv(
            "EMBEDDING_BASE_URL",
            DEFAULT_ONLINE_BASE_URL,
        )
        configured_model = model or os.getenv(
            "EMBEDDING_MODEL",
            DEFAULT_ONLINE_MODEL,
        )
        return cls(
            client=OpenAI(
                api_key=configured_api_key,
                base_url=configured_base_url,
            ),
            model=configured_model,
        )

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        normalized = _validate_texts(texts)
        if not normalized:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=normalized,
        )
        vectors = [list(item.embedding) for item in response.data]
        return _validate_vector_count(vectors, len(normalized))
