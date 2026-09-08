from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig

from src.langchaindemo.model import getModel

model = getModel()
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个聊天助手，用中文回答所有的问题"),
    HumanMessagePromptTemplate.from_template("{input}"),
])
parser = StrOutputParser()
chain = prompt | model | parser


def get_session_history(session_id):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=300
    )


# 串联历史记录
# 将对话历史自动集成到模型调用链中
# 和多用户支持中的核心问题
chat_bot_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input"
)

# # 模拟两个用户的会话
# # user1：
# resp_user1 = chat_bot_history.invoke(
#     input={"input": "我是user1，喜欢音乐，你能给我推荐一首轻音乐吗？"},
#     config={"configurable": {"session_id": "user1"}}
# )
# print("user1的回答：", resp_user1)
#
# # user2
# resp_user2 = chat_bot_history.invoke(
#     input={"input": "我是user2，热爱运动，篮球是我的最爱，你猜我喜欢哪个NBA球星。"},
#     config={"configurable": {"session_id": "user2"}}
# )
# print("user2的回答：", resp_user2)

# user1再次提问：
resp_user1 = chat_bot_history.invoke(
    input={"input": "我喜欢什么？"},
    config={"configurable": {"session_id": "user1"}}
)
print("user1的回答：", resp_user1)

# user2再次提问：
resp_user2 = chat_bot_history.invoke(
    input={"input": "我喜欢什么？"},
    config={"configurable": {"session_id": "user2"}}
)
print("user2的回答：", resp_user2)
