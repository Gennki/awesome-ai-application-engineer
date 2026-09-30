from typing import List

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from src.langchaindemo.model import getModel


class State(BaseModel):
    messages: List[AnyMessage] = Field(default_factory=list)


# 创建状态图构建器，传入状态类型
graph_builder = StateGraph(State)


# 定义聊天机器人节点函数
def chatbot(state: State):
    # 初始化模型
    llm = getModel()
    # 当 State 是 BaseModel 时，传入节点函数的 state 是一个 State 实例，应使用属性访问 state.messages，
    # 而不是 state['messages']（后者适用于 TypedDict）。
    print(f"聊天机器人节点接收状态：{state.messages}")
    message = llm.invoke(state.messages)
    print(f"大模型答复：{message.content}")
    return {"messages": [message.content]}


# 将聊天机器人节点添加到图中
graph_builder.add_node("chatbot", chatbot)

# 添加图的边(连接关系)
# 从开始节点连接到聊天机器人节点
graph_builder.add_edge(START, "chatbot")
# 从聊天机器人节点连接到结束节点
graph_builder.add_edge("chatbot", END)

# 编译图，使其可执行
graph = graph_builder.compile()

# 主交互循环
while True:
    try:
        user_input = input("User:")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        # 处理用户输入并获取助手回复
        """在 graph.invoke({"messages": [{"role": "user", "content": user_input}]}) 中，您传入的是一个字典列表。
        虽然Pydantic可能尝试将其转换为 AnyMessage，但更可靠的方式是显式使用 LangChain 的消息类（如 HumanMessage）。"""
        graph.invoke({"messages": [HumanMessage(content=user_input)]})
    except:
        print("系统发生错误，停止！")
        break
