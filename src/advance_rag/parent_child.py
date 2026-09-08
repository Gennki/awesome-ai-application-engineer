# 父子索引代码示例
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getEmbedding, getModel

model = getModel()
embedding_model = getEmbedding()

# 加载数据
loader = TextLoader(Path(__file__).with_name("deepseek.md"), encoding="utf-8")
docs = loader.load()

# 创建父子文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)

# 创建向量数据库，用于存储子文档块的向量数据
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embedding_model
)

# 创建父块文档存储对象
store = InMemoryStore()

# 创建父子文档检索器，使用这个检索器检索子块，并直接返回父文档块
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 5}  # 检索时获取相似度最高的子块文档个数
)

# 添加文档集,检索器会自动对文档进行分割
retriever.add_documents(docs)
print(f"主文档块的数：{len(list(store.yield_keys()))}")

# 测试，相似性搜索
# '''
# 这里我们通过向量数据库的similarity_search方法搜索出来的是与用户问题相关的子文档块的内容,
# 它会返回该子文档块所属的主文档块的全部内容:
# '''
# print("-----------------similarity_search-----------------")
# sub_docs = vectorstore.similarity_search("deepseek的应用场景", k=5)
# print([doc.page_content for doc in sub_docs])
#
# print("-----------------获取父文档-----------------")
# parent_docs = retriever.invoke("deepseek的应用场景")
# print([doc.page_content for doc in parent_docs])

prompt = ChatPromptTemplate.from_template('''
请根据下面给出的上下文来回答问题：
{context}

问题：{question}
''')
chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()
response = chain.invoke({"question": "deepseek的应用场景"})
print(response)
