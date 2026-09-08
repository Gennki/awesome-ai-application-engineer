"""
大模型默认没有记忆
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from src.langchaindemo.model import getModel

model = getModel()
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是人工智能助手"),
    HumanMessagePromptTemplate.from_template("{text}")
])
parser = StrOutputParser()
chain = prompt | model | parser

while True:
    user_input = input("请输入 'quit' 退出程序：")
    if user_input == "quit":
        print("程序结束")
        break
    else:
        print(chain.invoke({"text": user_input}))

# 请输入 'quit' 退出程序：我叫张三
# 你好，张三！很高兴认识你。有什么我可以帮你的吗？😊
# 请输入 'quit' 退出程序：我是谁
# 我暂时无法直接知道您是谁，因为我没有接入您的个人身份信息或对话历史以外的资料。不过，如果您愿意告诉我，我很乐意根据您提供的信息来称呼您，或者一起聊聊关于“身份”的有趣话题。 😊
