"""Chroma 向量数据库的轻量封装。"""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


DEFAULT_COLLECTION_NAME = "rag_demo"


@dataclass(slots=True)
class ChromaVectorStore:
    """封装 RAG 示例需要的 Chroma 写入与查询操作。

    Chroma 也可以自行调用嵌入模型，但本项目已经有统一的 ``Embedder``
    接口，因此这里显式接收向量。这样更容易看清向量化发生在流程的哪一步，
    也能保证文档和问题始终使用同一个嵌入模型。
    """

    collection: Collection

    @classmethod
    def in_memory(
        cls,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> "ChromaVectorStore":
        """创建仅在当前进程存活期间有效的 Chroma 集合。"""
        if not isinstance(collection_name, str):
            raise TypeError("collection_name must be a string")
        if not collection_name.strip():
            raise ValueError("collection_name must not be blank")

        client = chromadb.EphemeralClient()
        return cls.from_client(client, collection_name=collection_name)

    @classmethod
    def from_client(
        cls,
        client: ClientAPI,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> "ChromaVectorStore":
        """使用给定 Chroma 客户端创建或获取余弦距离集合。"""
        # embedding_function=None 表示向量由 src/embedding.py 生成，
        # Chroma 只负责保存向量并执行近邻检索。
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        return cls(collection=collection)

    def add_documents(
        self,
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """将文档片段及其向量批量写入集合。"""
        normalized_documents = self._validate_documents(documents)
        normalized_embeddings = [list(vector) for vector in embeddings]
        if len(normalized_documents) != len(normalized_embeddings):
            raise ValueError("documents and embeddings must have the same length")

        # 每个片段都需要唯一 ID；UUID 允许同一集合多次追加不同文档。
        ids = [str(uuid4()) for _ in normalized_documents]
        self.collection.add(
            ids=ids,
            documents=normalized_documents,
            embeddings=normalized_embeddings,
        )

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        n_results: int = 3,
    ) -> list[str]:
        """按相似度返回与查询向量最接近的文档片段。"""
        if isinstance(n_results, bool) or not isinstance(n_results, int):
            raise TypeError("n_results must be an integer")
        if n_results <= 0:
            raise ValueError("n_results must be greater than 0")

        vector = list(query_embedding)
        if not vector:
            raise ValueError("query_embedding must not be empty")

        result = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["documents"],
        )
        # Chroma 支持一次查询多个向量，所以 documents 的结构是
        # list[list[str]]；本方法一次只查询一个问题，取第一组即可。
        document_groups = result.get("documents")
        if not document_groups:
            return []
        return [document for document in document_groups[0] if document]

    @staticmethod
    def _validate_documents(documents: Sequence[str]) -> list[str]:
        if isinstance(documents, str) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of strings")

        normalized = list(documents)
        if not normalized:
            raise ValueError("documents must not be empty")
        if any(not isinstance(document, str) for document in normalized):
            raise TypeError("documents must contain only strings")
        if any(not document.strip() for document in normalized):
            raise ValueError("documents must not contain blank strings")
        return normalized
