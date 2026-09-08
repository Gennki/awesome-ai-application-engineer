"""演示读取文档、向量检索和 LLM 问答的完整 RAG 流程。"""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from chromadb.errors import ChromaError
from dotenv import load_dotenv
from ollama import RequestError, ResponseError
from openai import OpenAIError

if __package__:
    from src.rag import embedding as embedding_tools
    from .llm import LLMClient
    from .text_splitter import RecursiveTextSplitter
    from .vector_store import ChromaVectorStore
else:  # 支持直接执行 python src/main.py
    import embedding as embedding_tools
    from llm import LLMClient
    from text_splitter import RecursiveTextSplitter
    from vector_store import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT_PATH = PROJECT_ROOT / "data" / "rag_sample.txt"
DEFAULT_QUESTION = "番茄区在什么条件下会开始灌溉？"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RAGResult:
    """一次 RAG 流程中适合展示给学习者的关键结果。"""

    chunks: list[str]
    vector_dimension: int
    retrieved_documents: list[str]
    answer: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chroma + LLM 的完整 RAG 示例")
    parser.add_argument(
        "--document",
        type=Path,
        default=DEFAULT_DOCUMENT_PATH,
        help="作为知识库来源的 UTF-8 文本文件",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="需要根据知识库回答的问题",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="文本块大小")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="相邻文本块的重叠大小",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="从向量数据库召回的文本块数量",
    )
    return parser


def read_document(path: Path) -> str:
    """读取 UTF-8 文档，并尽早拒绝空知识库。"""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("document must not be blank")
    return text


def create_embedder() -> embedding_tools.Embedder:
    """创建当前启用的向量服务；默认使用线上模型。"""
    embedder: embedding_tools.Embedder = embedding_tools.OnlineEmbedder.from_env()
    # 切换到本地模型时，注释上一行并取消下一行注释。
    # embedder = embedding_tools.LocalEmbedder.from_env()
    return embedder


def build_rag_prompt(question: str, contexts: Sequence[str]) -> str:
    """把向量数据库召回的片段整理为约束清晰的 RAG 提示词。"""
    if not question.strip():
        raise ValueError("question must not be blank")
    if not contexts:
        raise ValueError("contexts must not be empty")

    # 给片段编号可以帮助观察召回内容，也方便模型在回答时区分资料来源。
    context_text = "\n\n".join(
        f"[片段 {index}]\n{context}"
        for index, context in enumerate(contexts, start=1)
    )
    return (
        "请仅根据下面的参考资料回答问题。"
        "如果资料中没有答案，请明确回答“根据现有资料无法确定”。\n\n"
        f"参考资料：\n{context_text}\n\n"
        f"问题：{question}"
    )


def run_rag_demo(
        *,
        text: str,
        question: str,
        splitter: RecursiveTextSplitter,
        embedder: embedding_tools.Embedder,
        vector_store: ChromaVectorStore,
        llm_client: LLMClient,
        top_k: int,
) -> RAGResult:
    """执行文档分块、入库、检索和大模型问答的完整流程。"""
    chunks = splitter.split(text)
    if not chunks:
        raise ValueError("text must not be blank")
    if not question.strip():
        raise ValueError("question must not be blank")

    # 索引阶段：先为所有文档片段生成向量，再把文本和向量一起写入 Chroma。
    chunk_embeddings = embedder.embed(chunks)
    vector_store.add_documents(chunks, chunk_embeddings)

    # 检索阶段：问题必须用同一个嵌入模型转换到相同的向量空间。
    question_embeddings = embedder.embed([question])
    if not question_embeddings:
        raise ValueError("embedding service returned no question vector")
    retrieved_documents = vector_store.query(
        question_embeddings[0],
        n_results=top_k,
    )

    if not retrieved_documents:
        raise ValueError("vector store returned no documents")

    # 生成阶段：LLM 只接收召回片段，而不是整篇长文档。
    prompt = build_rag_prompt(question, retrieved_documents)
    answer = llm_client.complete(prompt)
    vector_dimension = len(chunk_embeddings[0]) if chunk_embeddings else 0
    return RAGResult(
        chunks=chunks,
        vector_dimension=vector_dimension,
        retrieved_documents=retrieved_documents,
        answer=answer,
    )


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    try:
        text = read_document(args.document)
        result = run_rag_demo(
            text=text,
            question=args.question,
            splitter=RecursiveTextSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            ),
            embedder=create_embedder(),
            # 教学示例默认使用内存数据库，程序退出后不会遗留测试数据。
            vector_store=ChromaVectorStore.in_memory(),
            llm_client=LLMClient.from_env(),
            top_k=args.top_k,
        )
    except (
            ChromaError,
            ConnectionError,
            OSError,
            OpenAIError,
            RequestError,
            ResponseError,
            TypeError,
            ValueError,
    ) as error:
        # 日志只记录异常类型，避免第三方服务异常中可能携带的敏感配置泄漏。
        logger.error("rag_demo_failed error_type=%s", type(error).__name__)
        return 1

    print(f"文档分块数量: {len(result.chunks)}")
    print(f"向量维度: {result.vector_dimension}")
    print(f"召回片段数量: {len(result.retrieved_documents)}")
    for index, document in enumerate(result.retrieved_documents, start=1):
        print(f"\n\t[片段{index}]： {document}")
    print(f"\n\nLLM 响应: {result.answer}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
