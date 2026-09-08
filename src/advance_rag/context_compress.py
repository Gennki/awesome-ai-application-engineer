# Post-Retrieval后检索：上下文压缩示例代码
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter, \
    DocumentCompressorPipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from src.langchaindemo.model import getEmbedding, getModel

model = getModel()
embedding_model = getEmbedding()


# 格式化输出文档内容
def format_print_docs(docs):
    for i, doc in enumerate(docs):
        print("-" * 100)
        print(f"Document {i}:\n{doc.page_content}")


# 加载文档
loader = TextLoader(Path(__file__).with_name("deepseek.md"), encoding="utf-8")
docs = loader.load()

# 文档切片
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# 创建向量库
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)
# 创建向量检索器
retriever = vectorstore.as_retriever()

docs = retriever.invoke("deepseek遭遇的网络攻击和服务中断")
print("-------------------------压缩前-------------------------")
format_print_docs(docs)

print("-------------------------第一种：使用LLMChainExtractor压缩-------------------------")
"""
LLMChainExtractor压缩
利用大语言模型从检索到的文档中提取与查询相关的信息
它会将文档内容输入到LLM中，让LLM分析并提取出最相关的部分，从而实现文档的压缩。
需要调用大语言模型，计算成本较高，并且处理速度相对较慢。
"""
compressor = LLMChainExtractor.from_llm(model)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
compressed_docs = compression_retriever.invoke("deepseek遭遇的网络攻击和服务中断")
print("-------------------------压缩后-------------------------")
format_print_docs(compressed_docs)

print("-------------------------第二种：使用LLMChainFilter压缩-------------------------")
"""
LLMChainFilter 同样基于大语言模型
工作方式：
对检索到的文档进行过滤，只保留与查询相关的文档。
它会让LLM判断每个文档是否与查询相关，如果相关则保留，否则过滤掉。
相对LLMChainExtractor来说，计算成本可能稍低一些，因为它只是进行简单的过滤操作。同时，也能有效地筛选出相关文档。
仍然依赖于大语言模型的调用，计算成本和处理速度仍然是需要考虑的因素。
"""
compressor = LLMChainFilter.from_llm(model)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
compressed_docs = compression_retriever.invoke("deepseek遭遇的网络攻击和服务中断")
print("-------------------------压缩后-------------------------")
format_print_docs(compressed_docs)

print("-------------------------第三种：使用 EmbeddingFilter 压缩-------------------------")
"""
对每个检索到的文档进行额外的LLM调用既昂贵又缓慢.
EmbeddingFilter 通过嵌入文档和查询并仅返回那些与查询具有足够相似嵌入的文档来提供更便宜且更快的选项。
EmbeddingFilter 通过计算文档和查询的嵌入向量之间的相似度，返回与查询相似度超过设定阈值的文档
本质上就是利用嵌入模型将文档和查询转换为向量表示，然后使用余弦相似度等方法来衡量它们之间的相似性。
计算成本较低，处理速度较快，因为它主要是基于向量计算，而不需要调用大语言模型。
相似度的判断可能不够准确，因为它只基于嵌入向量的相似度，而没有考虑语义的深层次理解
"""
compressor = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.6)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
compressed_docs = compression_retriever.invoke("deepseek遭遇的网络攻击和服务中断")
print("-------------------------压缩后-------------------------")
format_print_docs(compressed_docs)

print("-------------------------第四种：组合压缩-------------------------")
"""
DocumentCompressorPipeline 轻松地按顺序组合多个压缩器
首先TextSplitters可以用作文档转换器，将文档分割成更小的块，
然后EmbeddingsRedundantFilter 根据文档之间嵌入的相似性来过滤掉冗余文档，
该过滤操作以文本的嵌入向量为依据，也就是借助余弦相似度来衡量文本之间的相似程度，
进而判定是否存在冗余，它会把文本列表转化成对应的嵌入向量，然后计算每对文本之间的余弦相似度。
一旦相似度超出设定的阈值，就会将其中一个文本判定为冗余并过滤掉。
最后 EmbeddingsFilter 根据与查询的相关性进行过滤。
"""
# 创建字符文本分割器，设置每个块的大小为300个字符，块之间无重叠，分隔符为句号
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator="。")
# 创建基于嵌入的冗余过滤器，使用之前获取的嵌入模型
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding_model)  # 去重，冗余
# 创建基于嵌入的相关性过滤器，使用之前获取的嵌入模型，设置相似度阈值为0.6
relevant_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.6)
# 创建文档压缩管道：先分割文档，然后过滤冗余文档，最后根据查询相关性过滤文档。
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever,
)
compressed_docs = compression_retriever.invoke("deepseek遭遇的网络攻击和服务中断")
print("-------------------------压缩后-------------------------")
format_print_docs(compressed_docs)

prompt = ChatPromptTemplate.from_template(
    "请根据上下文回答问题；上下文没有依据时请直接说不知道。\n"
    "上下文：{context}\n问题：{question}"
)
answer_chain = RunnableMap({
    "context": lambda x: "\n\n".join(doc.page_content for doc in compressed_docs),
    "question": lambda x: x["question"],
}) | prompt | model | StrOutputParser()
print("-------------------------压缩后的回答-------------------------")
print(answer_chain.invoke({"question": "deepseek遭遇的网络攻击和服务中断"}))
