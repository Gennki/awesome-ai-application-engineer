from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from src.langchaindemo.model import getModel

model = getModel()
prompt = PromptTemplate.from_template("用5句话来介绍{topic}")
parser = StrOutputParser()

# chain = prompt | model | parser
# 按顺序执行节点，效果同上
chain = RunnableSequence(prompt, model, parser)
print(chain.invoke({"topic": "人工智能"}))
