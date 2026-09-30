from typing import Literal

from langchain_core.messages import RemoveMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.langchaindemo.model import getModel


@tool
def web_search(query: str) -> str:
    """网络搜索"""
    # 代替实际实现
    return "上海阳光明媚"


tools = [web_search]
tool_node = ToolNode(tools)
model = getModel()
bound_model = model.bind_tools(tools)
memory = MemorySaver()


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}


# 没有工具调用时，先进入删除消息节点
def should_continue(state: MessagesState) -> Literal["action", "delete_messages"]:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "delete_messages"
    return "action"


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node(delete_messages)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("action", "agent")

# 删除消息后结束本轮执行
workflow.add_edge("delete_messages", END)
app = workflow.compile(checkpointer=memory)
app.get_graph().draw_mermaid_png(output_file_path="../../../../assets/删除消息节点.png")

config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="你好，我是张三")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print(event)
    print([(message.type, message.content) for message in event["messages"]])

input_message = HumanMessage(content="我叫什么名字？")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print([(message.type, message.content) for message in event["messages"]])
