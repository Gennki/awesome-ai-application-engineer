import os
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def getModel(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs

):
    return init_chat_model(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def getEmbedding(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv(
            "OPENAI_EMBEDDING_BASE_URL",
            os.getenv("EMBEDDING_BASE_URL"),
        ),
):
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
        chunk_size=20,
        max_retries=2,
        timeout=60
    )


class OpenRouterReranker(BaseDocumentCompressor):
    """基于 OpenRouter rerank 接口（Cohere 兼容格式）的重排序器。

    请求: POST {base_url}/rerank，body 为 {model, query, documents, top_n}
    响应: results 按相关度降序，每项含 index（对应输入文档下标）与 relevance_score

    继承 BaseDocumentCompressor，可直接配合 ContextualCompressionRetriever 使用：
        retriever = ContextualCompressionRetriever(
            base_compressor=getReranker(top_n=5), base_retriever=vector_retriever
        )
    """

    model: str
    api_key: str
    base_url: str
    top_n: int = 5

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        response = httpx.post(
            f"{self.base_url.rstrip('/')}/rerank",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": [doc.page_content for doc in documents],
                "top_n": self.top_n,
            },
            timeout=60,
        )
        response.raise_for_status()

        # results 按相关度降序返回，index 是文档在入参中的下标
        results = response.json()["results"]
        # 把相关度分数写入 metadata，便于下游按分数做阈值过滤
        return [
            documents[item["index"]].model_copy(
                update={"metadata": {
                    **documents[item["index"]].metadata,
                    "relevance_score": item["relevance_score"],
                }}
            )
            for item in results
        ]


def getReranker(
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        top_n: int = 5,
) -> OpenRouterReranker:
    """构建重排序器，配置默认取自 RERANK_* 环境变量

    :param model: 重排序模型名，默认 RERANK_MODEL（如 cohere/rerank-4-pro）
    :param api_key: API 密钥，默认 RERANK_API_KEY
    :param base_url: 接口地址，默认 RERANK_BASE_URL（OpenRouter 为 https://openrouter.ai/api/v1）
    :param top_n: 重排序后保留的文档数
    :return: 可配合 ContextualCompressionRetriever 使用的重排序器
    """
    resolved_model = model or os.getenv("RERANK_MODEL")
    resolved_api_key = api_key or os.getenv("RERANK_API_KEY")
    resolved_base_url = base_url or os.getenv("RERANK_BASE_URL", "https://openrouter.ai/api/v1")

    missing = [
        name for name, value in (
            ("RERANK_MODEL", resolved_model),
            ("RERANK_API_KEY", resolved_api_key),
            ("RERANK_BASE_URL", resolved_base_url),
        ) if not value
    ]
    if missing:
        raise ValueError(f"缺少重排序模型配置，请在 .env 中设置: {', '.join(missing)}")

    return OpenRouterReranker(
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        top_n=top_n,
    )
