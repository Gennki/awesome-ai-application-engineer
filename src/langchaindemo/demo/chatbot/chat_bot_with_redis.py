from langchain_community.chat_message_histories import RedisChatMessageHistory

from src.langchaindemo.model import getModel

model = getModel()

# pip install redis
history = RedisChatMessageHistory(
    session_id="my_session_id1",
    url="redis://localhost:6379",
    ttl=300  # 300秒过期
)

# 第一次对话
# history.add_user_message("你是谁？")
# ai_message = model.invoke(history.messages)
# history.add_ai_message(ai_message)
# print(ai_message)

# 第二次对话
history.add_user_message("重复一次")
ai_message = model.invoke(history.messages)
history.add_ai_message(ai_message)
print(ai_message)
