import bs4
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getEmbedding, getModel

# 获取网页内容
loader = WebBaseLoader(
    web_path="https://www.gov.cn/lianbo/202608/content_7078029.htm",
    # 分割。将网页中的目标内容进行分割
    bs_kwargs={"parse_only": bs4.SoupStrainer(id="UCAP-CONTENT")}
)
docs = loader.load()

# 文档切割
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
documents = splitter.split_documents(docs)

# 向量化
embedding = getEmbedding()
vector_store = Chroma.from_documents(documents, embedding)

# 检索器
retriever = vector_store.as_retriever()

system_prompt = """
你是一个问答任务助手。使用以下上下文来回答问题。
上下文：{context}

如果不知道答案，不要从其他渠道获取答案，直接说不知道。
"""
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{input}")
])

model = getModel()
# 创建文档链
chain1 = create_stuff_documents_chain(model, prompt_template)
# 创建检索连
chain2 = create_retrieval_chain(retriever, chain1)
response = chain2.invoke({"input": "发布会讲了什么事？"})
print(type(response))
print(response)
print("=" * 100)
print(response["answer"])
