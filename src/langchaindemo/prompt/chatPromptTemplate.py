from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from src.langchaindemo.model import getModel

model = getModel()

prompt = ChatPromptTemplate.from_messages([
    # ("system", "你是一个翻译助手，请将以下内容翻译成{language}"),
    SystemMessagePromptTemplate.from_template("你是一个翻译助手，请将以下内容翻译成{language}"),
    # ("human", "{text}"),
    HumanMessagePromptTemplate.from_template("{text}")
    # AIMessagePromptTemplate ai的角色信息，用来记录维持对话
])
fact_prompt = prompt.format(language="中文", text="I am a programmer")
print(fact_prompt)
result = model.invoke(fact_prompt)
print(result.content)
