"""
create_stuff_documents_chain 用于把一组 Document 文档内容合并后塞进 Prompt，再交给大模型回答问题。
"""
import bs4
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getModel

model = getModel()
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("根据提供的上下文：{context} \n\n 回答问题"),
    HumanMessagePromptTemplate.from_template("问题：{input}")
])
chain = create_stuff_documents_chain(model, prompt)

# 加载文档
loader = WebBaseLoader(
    web_path="https://www.gov.cn/lianbo/202608/content_7078029.htm",
    # 分割。将网页中的目标内容进行分割
    bs_kwargs={"parse_only": bs4.SoupStrainer(id="UCAP-CONTENT")}
)
docs = loader.load()

# 分割文档
text_spliter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
documents = text_spliter.split_documents(docs)
print(len(documents))

# 执行链，检索“文章具体讲了什么事？”
print(chain.invoke({"input": "文章具体讲了什么事？", "context": documents[0:5]}))
