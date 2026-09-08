# 作用：可以给提示词模板分开赋值

from langchain_core.prompts import PromptTemplate

from src.langchaindemo.model import getModel

model = getModel()

prompt = PromptTemplate.from_template("今天是{date}，讲一个关于{type}的小故事。")
half_prompt = prompt.partial(date="2026-08-11")
print(half_prompt)
result = model.invoke(half_prompt.format(type="妖魔"))
print(result.content)
