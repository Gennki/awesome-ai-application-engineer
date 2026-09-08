from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.langchaindemo.model import getModel

model = getModel()
model.streaming = True

prompt = PromptTemplate.from_template("用5句话来介绍{topic}")
parser = StrOutputParser()

chain = prompt | model | parser
# print(chain.invoke({"topic": "人工智能"}))
for chunk in chain.stream({"topic": "人工智能"}):
    print(chunk, end="", flush=True)
