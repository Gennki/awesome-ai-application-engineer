from langchain_core.tools import tool

from src.langchaindemo.model import getModel


@tool
def multiply(a: int, b: int) -> int:
    """计算两数相乘."""
    return a * b

from dotenv import load_dotenv
load_dotenv()
llm = getModel()

# #使用工具，需要和模型进行绑定
# model_with_tools = llm.bind_tools([multiply])
# response_message = model_with_tools.invoke("42 x 7等于多少?")
# tool_call = response_message.tool_calls[0]
# print("大模型答复：",tool_call)
# print("工具执行：",multiply.invoke(tool_call))

#LangGrahp中专门定义了一个ToolNode，作为工具节点
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

@tool
def add(a: int, b: int) -> int:
    """计算两数相加."""
    return a + b

tool_node = ToolNode([multiply,add])

model_with_tools = llm.bind_tools([multiply,add])

def should_continue(state: MessagesState):

    messages = state["messages"]
    print("messages:", messages)
    last_message = messages[-1]
    print("last_message:",last_message)
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

graph = builder.compile()
print(graph.invoke({"messages": [{"role": "user", "content": "1000+234=?"}]}))
print(graph.invoke({"messages": [{"role": "user", "content": "875*234=?"}]}))