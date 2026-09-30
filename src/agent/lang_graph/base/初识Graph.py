from typing import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.langchaindemo.model import getModel

"""
需求：需要基于LangGraph构建一张图、并且需要让这张图有与AI对话的能力
需求分析：
第一步：首先我需要有一张空白的图--------->构建一张图 「类和对象的关系」
from langgraph.graph import StateGraph
graph_builder = StateGraph(State)

第二步：构建状态-------->对于这张图来说，我可以接收哪些数据类型------->TypedDict「可以对相关性的数据类型进行声明」
class State(TypedDict):
    messages: list[AnyMessage]

第三步：构建相关性的Nodes与Edges
def chatbot(state: State):
    pass

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

第四步：编译图
graph = graph_builder.compile()


"""


# 定义状态类型，使用TypedDict来明确状态的结构
class State(TypedDict):
    # 消息列表，使用Annotated添加元数据（这里指定了消息处理方式）
    messages: list[AnyMessage]


# 创建状态图构建器，传入状态类型
graph_builder = StateGraph(State)


# 定义聊天机器人节点函数
def chatbot(state: State):
    # 初始化模型
    llm = getModel()
    print(f"聊天机器人节点接收状态：{state["messages"]}")
    message = llm.invoke(state["messages"])
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
        graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    except:
        print("系统发生错误，停止！")
        break
