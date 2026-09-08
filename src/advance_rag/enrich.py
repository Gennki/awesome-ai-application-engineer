# 预检索相关，Enrich完善问题代码示例
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory

from src.langchaindemo.model import getModel, getEmbedding

model = getModel()
embedding_model = getEmbedding()

# 示例业务模板
templates = {
    "订机票": ["起点", "终点", "时间", "座位等级", "座位偏好"],
    "订酒店": ["城市", "入住日期", "退房日期", "房型", "人数"],
}
print(list(templates.keys()))

# 意图识别提示词模板
intent_prompt = PromptTemplate(
    input_variables=["user_input", "templates"],
    template="根据用户输入 '{user_input}'，选择最合适的业务模板。可用业务模板如下：{templates}。请返回模板名称"
)

# 创建意图识别链
intent_chain = intent_prompt | model

# 模拟用户输入
user_input = input("请输入问题：")
# 识别用户意图
intent = intent_chain.invoke({"user_input": user_input, "templates": list(templates.keys())}).content
print("意图：", intent)
# 获取对应模板
selected_template = templates.get(intent)
print("模板：", selected_template)

# 构造信息补全Prompt
info_prompt = f"""
请根据用户原始问题和模板，判断问题是否完善。如果问题缺乏需要的信息，请生成一个友好的请求，明确指出需要补充的信息。
若完善后，返回包含所有信息的完整问题。

### 原始问题：
{user_input}

### 模板：
{selected_template}

### 输出示例：
{{
    "isComplete": true,
    "content":"完整问题"
}}
{{
    "isComplete": false,
    "content":"友好的引导要补充的信息"
}}
"""

# 创建聊天模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个信息补充助手，任务是分析用户问题是否完整。"),
    ("placeholder", "{history}"),  # 历史记录的占位
    ("human", "{input}")
])

# 创建信息补全链
info_chain = prompt | model

# 自动处理历史记录，将记录注入输入并在每次调用后更新它
history = InMemoryChatMessageHistory()
with_message_history = RunnableWithMessageHistory(
    info_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

# 判断问题是否完整，如果不完整，则生成追问请求
info_request = with_message_history.invoke(input={"input": info_prompt},
                                           config={"configurable": {"session_id": "unused"}}).content
parser = JsonOutputParser()
json_data = parser.parse(info_request)

while json_data.get("isComplete", False) is False:
    user_answer = input(f"{json_data.get('content')}\n\n你的回复：")
    # 提交补充信息给AI处理
    info_request = with_message_history.invoke(
        input={"input": user_answer},
        config={"configurable": {"session_id": "unused"}}
    ).content
    json_data = parser.parse(info_request)

# 输出最终结果
print(f"[最终查询]{info_request}")
