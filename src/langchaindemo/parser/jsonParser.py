"""JSON解析器,确保输出符合特定JSON对象格式"""

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from src.langchaindemo.model import getModel

model = getModel()

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个专业的程序员"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

parser = JsonOutputParser()
chain = prompt | model | parser
result = chain.invoke({"input": "langchain是什么？问题用用question,回答用answer，返回一个JSON格式"})
print(result)
