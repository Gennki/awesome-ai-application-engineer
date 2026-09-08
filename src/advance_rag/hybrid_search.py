# 混合检索代码示例
import jieba
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getModel, getEmbedding

model = getModel()
embedding_model = getEmbedding()


# 格式化输出文档内容
def format_print_docs(docs):
    for i, doc in enumerate(docs):
        print("-" * 100)
        print(f"Document {i}:\n{doc.page_content}")


# 加载文档
loader = TextLoader("deepseek.md", encoding="utf-8")
docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 创建向量存储
vectorstore = Chroma.from_documents(split_docs, embedding=embedding_model)

question = "相关事件"

# 向量检索
vectorstore_receiver = vectorstore.as_retriever(search_kwargs={"k": 3})
vectorstore_docs = vectorstore_receiver.invoke(question)
print("--------------------向量检索--------------------")
format_print_docs(vectorstore_docs)

# 关键词检索
# BM25 默认按空格分词，中文没有空格会导致整段变成一个"词"、检索完全失效，改用 jieba 分词
def jieba_split(text):
    return [t for t in jieba.cut_for_search(text) if t.strip()]


bm25_retriever = BM25Retriever.from_documents(split_docs, preprocess_func=jieba_split)
bm25_retriever.k = 3
bm25_docs = bm25_retriever.invoke(question)
print("--------------------BM25检索--------------------")
format_print_docs(bm25_docs)

# 混合检索 EnsembleRetriever 混合检索器
# EnsembleRetriever 是 LangChain 集合多个检索器的检索器。 weights 1 表示权重
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vectorstore_receiver],
    weights=[0.3, 0.7]
)
ensemble_docs = ensemble_retriever.invoke(question)
print("--------------------混合检索--------------------")
format_print_docs(ensemble_docs)

# 创建提示词模板
prompt = ChatPromptTemplate.from_template("""
请根据下面给出的上下文来回答问题：
{context}
问题：{question}
""")

# 创建2个执行链进行对比
chain1 = RunnableMap({
    "context": lambda x: ensemble_retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()

chain2 = RunnableMap({
    "context": lambda x: vectorstore_receiver.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()

print("--------------------模型回复--------------------")
print("--------------------混合检索--------------------")
print(chain1.invoke({"question": question}))
print("--------------------向量检索--------------------")
print(chain2.invoke({"question": question}))
