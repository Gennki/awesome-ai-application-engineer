# 安装 pip install langchain_chroma
# 加载 word 文档 安装 pip install docx2txt
# 加载 json 文档 安装 pip install jq
# 加载 pdf 文档 安装 pip install pymupdf
# 加载 HTML 文档 安装 pip install unstructured
# 加载 MD 文档 安装 pip install markdown + pip install unstructured
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getEmbedding, getModel

model = getModel()

# 加载文档，转换位Document格式
document_path = Path(__file__).with_name("RAG系统工程与评测实战手册_复杂知识库样本.docx")
loader = Docx2txtLoader(str(document_path))
documents = loader.load()

# 切割文档
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)
split_documents = text_spliter.split_documents(documents)

# 实例化向量空间，向量化+向量存储到向量数据库中
llm_embedding = getEmbedding()
vector_store = Chroma.from_documents(documents=split_documents, embedding=llm_embedding)

# 获取检索器
# 使用向量相似度检索，默认top-k=4
# retriever = vector_store.as_retriever()

# 使用向量相似度阈值检索
# retriever = vector_store.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 4,
#         "score_threshold": 0.4
#     },
# )

# 使用mmr(最大边际相似度)检索，一种平衡相关性和多样性的检索方式
# 应用场景：知识库中有大量相似片段时，既要召回相关内容，又希望结果覆盖不同角度，避免重复信息占满上下文。
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.5,  # MMR 返回结果的多样性；`1` 表示多样性最低，`0` 表示多样性最高。（默认值：`0.5`）
    },
)
# result = retriever.invoke("评测集设计")
# print(result)

message = """
仅使用提供的上下文回答下面的问题：
{question}

上下文：
{context}
"""
prompt_template = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(message)])
chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(
            lambda docs: "\n\n".join(doc.page_content for doc in docs)
        ),
    }
    | prompt_template
    | model
    | StrOutputParser()
)
print(chain.invoke("评测集设计"))
