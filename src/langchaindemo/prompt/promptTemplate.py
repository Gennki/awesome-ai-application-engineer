from langchain_core.prompts import PromptTemplate
from src.langchaindemo.model import getModel

model = getModel()

prompt = PromptTemplate.from_template("你是一个翻译助手，请将以下内容翻译成{language}:{text}")
fact_prompt = prompt.format(language="中文", text="I am a programmer")
print(fact_prompt)
result = model.invoke(fact_prompt)
print(result.content)
