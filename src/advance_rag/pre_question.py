# 假设性问题索引示例代码
import uuid
from typing import List

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.stores import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from src.langchaindemo.model import getModel, getEmbedding

model = getModel()
embedding_model = getEmbedding()

# 加载文档
loader = TextLoader("deepseek.md")
docs = loader.load()

# 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# 初始化Chroma向量数据库，存储生成的问题向量
vectorstore = Chroma(
    collection_name="pre-questions",
    embedding_function=embedding_model
)

# 初始化内存存储，存储原始文档块
store = InMemoryByteStore()
id_key = "doc_id"

# 设置多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key
)

# 为每个原始文档生成唯一id
doc_ids = [str(uuid.uuid4()) for _ in docs]


# 以下开始用大模型生成假设性问题
class HypotheticalQuestions(BaseModel):
    """约束生成假设性问题的格式"""
    questions: List[str] = Field(..., description="List of questions")


# 此处使用{{}}是为了避免ChatPromptTemplate按f-string语法解析成模板变量:{变量名称},因此需要转义
prompt = ChatPromptTemplate.from_template("""
请基于以下文档生成3个假设性问题（必须使用JSON格式）：
{doc}

要求：
1. 输出必须为合法JSON格式，包含questions字段
2. questions字段的值是包含3个问题的数组
3. 使用中文提问
示例格式：
{{
    "questions":[
        "问题1",
        "问题2",
        "问题3"
    ]
}}

""")

# 创建假设性问题链
"""
其中的model.with_structured_output可以理解为输出解析器的一种更高级用法
将大模型的输出转换为HypotheticalQuestions所限定的格式,
而HypotheticalQuestions要求的格式是:
定义了一个字段questions,它具有以下特性:
类型注解:List[str]表示questions字段应该是一个字符串列表。
必需性:Field(...)中的省略号...表示这个字段是必需的
描述信息:description="List of questions"为该字段添加了描述,这对于生成文档或帮助理解模型结构很有用。

注意：这里必须通过 extra_body 关闭思考模式，原因如下：
- deepseek-v4-flash 默认开启思考模式（thinking 参数默认值为 enabled），
  而思考模式下 DeepSeek API 不支持强制指定 tool_choice；
- with_structured_output 默认的 method="function_calling" 会把 Pydantic schema
  包装成 tool 并强制 tool_choice 指定该工具，恰好触发 DeepSeek 的限制，
  报错：Thinking mode does not support this tool_choice；
- 解决思路：关闭思考模式后该限制即解除，可以继续使用默认的 function_calling
  方式（服务端强制模型调用 tool，输出结构由服务端保证，比 json_mode 的
  客户端解析更可靠，且换其他支持 function calling 的模型时同样适用）；
- extra_body 的作用：openai SDK 的请求体里没有现成的 thinking 字段
  （它是 DeepSeek 的私有参数，不是 OpenAI 标准参数），extra_body 中的键值
  会被原样合并进 JSON 请求体，专门用于透传这类厂商私有参数；
- extra_body 只通过 kwargs 绑定到这一条 structured-output 链上，
  不会影响 getModel() 返回的原始 model 对象在其他地方的用法。
"""
chain = ({"doc": lambda x: x.page_content} |
         prompt |
         # 将LLM输出构建为字符串列表（思考模式不支持强制tool_choice，先关闭思考模式）
         model.with_structured_output(
             HypotheticalQuestions,
             extra_body={"thinking": {"type": "disabled"}}
         ) |
         # 提取问题列表
         (lambda x: x.questions)
         )

# # 测试：在单个文档上调用链，链的最终输出是大模型答复的假设性问题列表
# print("测试：", docs[0])
# print("测试生成的问题：", chain.invoke(docs[0]))

# 批量处理所有文档生成假设性问题（设置最大并行数5），每个切块后的文档块，都对应生成三个问题
hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})
# print("假设性问题列表：", hypothetical_questions)

# 将生成的问题转换为带元数据的文档对象
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list
    )
# print(question_docs)

# 将问题文档存入向量数据库
retriever.vectorstore.add_documents(question_docs)
# 将原始文档存入字节存储
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 测试：执行相似性搜索
query = "deepseek受到哪些攻击？"
retriever_docs = retriever.invoke(query)
print("------------------------检索到的问题------------------------")
print(retriever_docs)

prompt = ChatPromptTemplate.from_template("根据下面的文档回答问题：\n\n{doc}\n\n问题：{question}")
chain = RunnableMap({
    "doc": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()
answer = chain.invoke({"question": query})
print("------------------------回答------------------------")
print(answer)
