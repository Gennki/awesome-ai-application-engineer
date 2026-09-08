"""CSV解析器,输出以逗号分隔、列表形式返回"""
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from src.langchaindemo.model import getModel

model = getModel()
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个专业的程序员"),
    HumanMessagePromptTemplate.from_template("{input}")
])

parser = CommaSeparatedListOutputParser()
chain = prompt | model | parser
print(chain.invoke({"input": "列出Python的三个主要版本，用逗号分隔"}))
print(chain.invoke({"input": "列举三个常见的机器学习框架，用逗号分隔"}))
