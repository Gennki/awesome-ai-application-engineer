# 多路召回代码示例
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getEmbedding, getModel

model = getModel()

# 加载文档
loader = TextLoader("deepseek.md", encoding="utf-8")
docs = loader.load()

# 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# 创建向量数据库
vectorstore = Chroma.from_documents(documents=documents, embedding=getEmbedding())

# 创建检索器
retriever = vectorstore.as_retriever()

print("--------------------原始检索--------------------")
question = "deepseek的应用场景"
relevant_docs = retriever.invoke(question)
print(relevant_docs)
# 查看一下检索到的相关文档的数量
print("检索器检索的文档数量为：", len(relevant_docs))

# 创建prompt模板
prompt = ChatPromptTemplate.from_template("""
请根据下面给出的上下文来回答问题：
{context}

问题：{question}
""")

print("--------------------优化前回答--------------------")
chain = RunnableMap({
    "context": lambda x: relevant_docs,
    "question": lambda x: x["question"],
}) | prompt | model | StrOutputParser()

print(chain.invoke({"question": question}))

print("--------------------开始优化检索--------------------")
# 使用langchain的MultiQueryRetriever
# 引入日志组件查看llm在原查询的基础上生成的多个查询
import logging

logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

# 定义多路检索器的prompt
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model
)
# invoke 的入参是字符串（不是 dict），LLM 会基于它生成多个不同视角的查询
unique_docs = retriever_from_llm.invoke(question)
print(unique_docs)
print("检索器检索的文档数量为：", len(unique_docs))

print("--------------------优化后回答--------------------")
final_rag_chain = (
        {
            "context": lambda x: unique_docs,
            "question": lambda x: x["question"]
        } | prompt | model | StrOutputParser()
)

print(final_rag_chain.invoke({"question": question}))
