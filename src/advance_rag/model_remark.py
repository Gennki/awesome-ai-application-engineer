# 后检索优化——重排序（Rerank）代码示例
# 重排序器的实现见 src/langchaindemo/model.py 中的 OpenRouterReranker
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getEmbedding, getModel, getReranker

model = getModel()
embedding_model = getEmbedding()


# 格式化输出文档内容（带相关度分数）
def format_print_docs(docs):
    for i, doc in enumerate(docs):
        score = doc.metadata.get("relevance_score", "无（重排序前）")
        print("-" * 100)
        print(f"Document {i}（相关度分数={score}）:\n{doc.page_content}")


# 加载文档
loader = TextLoader(Path(__file__).with_name("deepseek.md"), encoding="utf-8")
docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 创建向量数据库
vectorstore = Chroma.from_documents(split_docs, embedding=embedding_model)

question = "deepseek的应用场景"

# 向量检索
# k 故意调大到6：向量检索负责"捞得全"（宁可多召回也不要漏掉），
# 捞回来的候选集排序准不准，交给后面的重排序负责
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

print("--------------------向量检索（召回6个候选文档）--------------------")
retrieved_docs = vector_retriever.invoke(question)
print("向量检索返回的文档数量为：", len(retrieved_docs))
format_print_docs(retrieved_docs)

print("--------------------模型重排序--------------------")
# 重排序模型（如 Cohere rerank）是专门的"精排"模型：
# 它把查询和每个文档放在一起做深度语义匹配，逐个给出相关度分数，
# 按分数降序输出，排序质量远高于向量相似度
# top_n=3：重排序后只保留相关度最高的3个文档（多路召回、精排收口）
reranker = getReranker(top_n=3)

# 直接调用重排序器：传入候选文档和查询，输出按相关度降序排列的文档
# 相关度分数会写入每个文档的metadata["relevance_score"]
reranked_docs = reranker.compress_documents(retrieved_docs, question)
print("重排序后保留的文档数量为：", len(reranked_docs))
format_print_docs(reranked_docs)

print("--------------------基于分数的阈值过滤--------------------")
# 除了按top_n截断，还可以按分数阈值过滤：只保留高置信度的文档，
# 相关但置信度不高的文档宁可丢弃，避免干扰大模型生成答案
THRESHOLD = 0.3
for doc in reranked_docs:
    score = doc.metadata["relevance_score"]
    mark = "保留" if score >= THRESHOLD else "丢弃"
    print(f"分数={score:.4f} → {mark} | {doc.page_content.strip()[:50]}")
threshold_docs = [doc for doc in reranked_docs if doc.metadata["relevance_score"] >= THRESHOLD]

print("--------------------ContextualCompressionRetriever组合使用--------------------")
# 上面的"检索→重排序"两步也可以用ContextualCompressionRetriever一步完成：
# base_retriever负责召回候选，base_compressor负责精排压缩
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vector_retriever,
)
compressed_docs = compression_retriever.invoke(question)
print("组合检索器返回的文档数量为：", len(compressed_docs))
format_print_docs(compressed_docs)

# 创建提示词模板
prompt = ChatPromptTemplate.from_template("""
请根据下面给出的上下文来回答问题：
{context}
问题：{question}
""")

# 创建2个执行链进行对比
chain1 = RunnableMap({
    "context": lambda x: "\n\n".join(doc.page_content for doc in retrieved_docs),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()

chain2 = RunnableMap({
    "context": lambda x: "\n\n".join(doc.page_content for doc in compressed_docs),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()

chain_threshold = RunnableMap({
    "context": lambda x: "\n\n".join(doc.page_content for doc in threshold_docs),
    "question": lambda x: x["question"],
}) | prompt | model | StrOutputParser()

print("--------------------向量检索（6个候选文档直接回答）--------------------")
print(chain1.invoke({"question": question}))
print("--------------------重排序后的阈值过滤回答--------------------")
if threshold_docs:
    print(chain_threshold.invoke({"question": question}))
else:
    print("没有文档达到当前阈值，跳过回答。")
print("--------------------组合检索器回答--------------------")
print(chain2.invoke({"question": question}))
