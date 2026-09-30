from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from src.langchaindemo.model import getModel

load_dotenv()

@tool
def weather_search(city: str):
    """搜索天气"""
    print("weather_search begin----")
    print(f"搜索: {city}")
    return "晴天"

#将工具绑定
model = getModel().bind_tools(
    [weather_search]
)

class State(MessagesState):
    """Simple state."""

def call_llm(state):
    print(f"大模型节点收到：{state}")
    return {"messages": [model.invoke(state["messages"])]}

def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[-1]
    human_review = interrupt(
        {
            "question": "正确吗？",
            "tool_call": tool_call,
        }
    )
    review_action = human_review["action"]
    review_data = human_review.get("data")
    # 如果获得批准，调用该工具
    if review_action == "continue":
        return Command(goto="run_tool")
    # update the AI message AND call tools
    elif review_action == "update":
        updated_message = {
            "role": "ai",
            "content": last_message.content,
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "args": review_data,
                }
            ],
            "id": last_message.id,
        }
        return Command(goto="run_tool", update={"messages": [updated_message]})
    elif review_action == "feedback":
        tool_message = {
            "role": "tool",
            "content": review_data,
            "name": tool_call["name"],
            "tool_call_id": tool_call["id"],
        }
        return Command(goto="call_llm", update={"messages": [tool_message]})

def run_tool(state):
    print(f"执行工具节点收到：{state}")
    new_messages = []
    tools = {"weather_search": weather_search}
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        new_messages.append(
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": result,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": new_messages}

def route_after_llm(state) -> Literal[END, "human_review_node"]:
    print(f"路由函数收到：{state}")
    if len(state["messages"][-1].tool_calls) == 0:
        return END
    else:
        return "human_review_node"


builder = StateGraph(State)
builder.add_node(call_llm)
builder.add_node(run_tool)
builder.add_node(human_review_node)
builder.add_edge(START, "call_llm")
builder.add_conditional_edges("call_llm", route_after_llm)
builder.add_edge("run_tool", "call_llm")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

graph.get_graph().draw_mermaid_png(output_file_path='../../../../../assets/审查工具使用.png')

#无需工具调用
initial_input = {"messages": [{"role": "user", "content": "你好!"}]}

# Thread
thread = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print(event)
    print("\n")


#进行工具调用
initial_input = {"messages": [{"role": "user", "content": "上海的天气如何?"}]}
for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print(event)
    print("\n")

print("等待人工审核中.........")
# 正在等待人工审核
print(graph.get_state(thread).next)

# # 1、同意继续执行
# for event in graph.stream(
#     Command(resume={"action": "continue"}),
#     thread,
#     stream_mode="updates",
# ):
#     print(event)
#     print("\n")

# # 2、手动修改工具调用然后继续
# for event in graph.stream(
#     Command(resume={"action": "update", "data": {"city": "上海，中国"}}),
#     thread,
#     stream_mode="updates",
# ):
#     print(event)
#     print("\n")

# 3、提供自然语言反馈
for event in graph.stream(
    Command(resume={"action": "feedback", "data": {"feedback": "根据最新天气信息，上海开始下雨了。"}}),
    thread,
    stream_mode="updates",
):
    print(event)
    print("\n")