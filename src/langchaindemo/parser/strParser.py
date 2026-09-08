"""StrOutputParser解析器,输出字符串"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.langchaindemo.model import getModel

model = getModel()

prompt = PromptTemplate.from_template("你是一个翻译助手，请将以下内容翻译成{language}:{text}")
# fact_prompt = prompt.format(language="中文", text="I am a programmer")
# print(fact_prompt)
# result = model.invoke(fact_prompt)
# # print(result.content)
#
# # 字符串解析器，将大模型返回的内容，获取到字符串内容
# parser = StrOutputParser()
# print(parser.invoke(result))

# 上述代码可以改为下面的chain的方式调用
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({"language": "中文", "text": "I am a programmer"})
print(result)
