"""日期时间解析器,可用于将LLM输出解析为日期时间格式"""
from langchain_classic.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate

from src.langchaindemo.model import getModel

model = getModel()
parser = DatetimeOutputParser()

template = """
回答用户的问题：{question}

{format_instructions}
"""
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser
result = chain.invoke({"question": "新中国是什么时候成立的？"})
print(result)
