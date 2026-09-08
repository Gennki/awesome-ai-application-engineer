import uvicorn
from fastapi import FastAPI
from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langserve import add_routes

from src.langchaindemo.model import getModel

# 开启LangChain应用程序的调试功能
set_debug(True)

model = getModel()
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("请将以下内容翻译成{language}"),
    HumanMessagePromptTemplate.from_template("{input}")
])
parser = StrOutputParser()
# 以链的形式调用
chain = prompt | model | parser
# 本地调用
# result = chain.invoke({"language": "中文", "input": "I am a developer"})
# print(result)
app = FastAPI(title="基于LangChain的服务", version="v1.0.0", description="翻译服务")
add_routes(app, chain, path="/langchainServer")

if __name__ == '__main__':
    """
    POST请求，请求Body体如下：
    {
        "input": {
            "language": "英文",
            "input": "今天天气不错。"
        }
    }
    最外层的“input”是固定写法
    """
    uvicorn.run(app, host="localhost", port=8000)
