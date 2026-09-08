from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder

from src.langchaindemo.model import getModel

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是人工智能助手"),
    # 作用就是向提示词中插入一段上下文消息
    # {"placeholder":"{messages}"}
    MessagesPlaceholder(variable_name="messages")
])
model = getModel()
parser = StrOutputParser()
chain = prompt | model | parser

# 创建消息历史记录，存储组件
chat_history = InMemoryChatMessageHistory()

while True:
    user_input = input("用户：")
    if user_input == "exit":
        break

    # 添加用户输入
    chat_history.add_user_message(user_input)
    # 访问LLM时，chat_history_messages获取所有的历史消息
    response = chain.invoke({"messages": chat_history.messages})

    print("chat_history:", chat_history.messages)
    print("大模型回复：", response)

    # 将大模型的回复加入历史记录
    chat_history.add_ai_message(response)


# 用户：我是张三
# chat_history: [HumanMessage(content='我是张三', additional_kwargs={}, response_metadata={})]
# 大模型回复： 你好，张三！很高兴认识你。有什么我可以帮你的吗？
# 用户：我是谁
# chat_history: [HumanMessage(content='我是张三', additional_kwargs={}, response_metadata={}), AIMessage(content='你好，张三！很高兴认识你。有什么我可以帮你的吗？', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='我是谁', additional_kwargs={}, response_metadata={})]
# 大模型回复： 根据我们刚才的聊天记录，你亲口告诉我你叫**张三**，所以在这个对话里，我一直把你当作张三。
#
# 当然，如果你想纠正一下（比如刚才报的是化名），或者想聊聊哲学意义上“我是谁”这种深奥的问题，我也很乐意陪你探讨！😄