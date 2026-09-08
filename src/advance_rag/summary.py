# 摘要索引示例代码
import uuid
from pathlib import Path

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.stores import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getEmbedding, getModel

model = getModel()

# 1. 提取、分块
# 加载文档
loader = TextLoader(Path(__file__).with_name("deepseek.md"), encoding="utf-8")
docs = loader.load()
# 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# 2. 将分块内容生成摘要
chain = {"doc": lambda x: x.page_content} | ChatPromptTemplate.from_template(
    "总结下面的文档：\n\n{doc}") | model | StrOutputParser()
print("准备生成文档摘要，请耐心等待...")
# 批量生成文本摘要，最大并发数设置5
summaries = chain.batch(docs, {"max_concurrency": 5})
print(summaries)

# 3. 摘要向量化，并与原始块建立映射关系
# 初始化Chroma示例，存储摘要向量
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=getEmbedding()
)
# 初始化内存字节存储，用于存储原始文档
store = InMemoryByteStore()
# 初始化多向量检索器，结合向量存储和文档存储
id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key
)
# 为每个文档生成唯一ID，该ID用于关联原始文档和摘要
doc_ids = [str(uuid.uuid4()) for _ in docs]
# 将文档摘要转换为LangChain中的Document
summary_doc = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# 4. 摘要向量存储到向量数据库，原始文档存到传统数据库
print("准备将摘要添加到向量数据库...")
retriever.vectorstore.add_documents(summary_doc)
print("准备将原始文档存储到字节存储...")
# mset：批量设置键值对
# list(zip(doc_ids,docs))：将ID和文档配对
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 5. 检索，匹配摘要向量，并返回原始文档
query = "deepseek的企业事件"

# 检索匹配摘要并通过摘要id获取原始文档
# sub_docs = retriever.vectorstore.similarity_search(query)
# print("--------------------------匹配的摘要内容--------------------------")
# print(*sub_docs, sep="\n")
#
# # 获取第一个匹配摘要的id
# matched_id = sub_docs[0].metadata[id_key]
# # 通过摘要id获取原始文档
# original_doc = retriever.docstore.mget([matched_id])
# print(original_doc)

# retriever.invoke(query)将直接获得关联的原始文档
retriever_docs = retriever.invoke(query)
print("--------------------------检索到的文档--------------------------")
print(retriever_docs)

prompt = ChatPromptTemplate.from_template("根据下面的文档回答问题：\n\n{doc}\n\n问题：{question}")
chain = RunnableMap({
    "doc": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()

query = "deepseek的企业事件"
answer = chain.invoke({"question": query})
print("--------------------------回答--------------------------")
print(answer)
