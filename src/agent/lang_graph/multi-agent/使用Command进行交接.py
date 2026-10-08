from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command
from typing_extensions import Literal

from src.langchaindemo.model import getModel

load_dotenv()

model = getModel()


@tool
def transfer_to_multiplication_expert():
    """向乘法专家求助."""
    # 这个工具不会返回任何东西:我们只是在使用它
    # 作为LLM发出需要移交给另一个代理的信号的一种方式
    return


@tool
def transfer_to_addition_expert():
    """向加法专家寻求帮助."""
    return


def addition_expert(state: MessagesState, ) -> Command[Literal["multiplication_expert", "__end__"]]:
    system_prompt = (
        "您是加法专家，您可以向乘法专家寻求乘法方面的帮助。 "
        "交接之前务必做好自己的那部分计算。"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    # 让 LLM 决定是否需要调用乘法专家的工具
    ai_msg = model.bind_tools([transfer_to_multiplication_expert]).invoke(messages)
    print(f"addition_expert:{ai_msg}")
    if len(ai_msg.tool_calls) > 0:  # 如果 LLM 决定需要交接
        tool_call_id = ai_msg.tool_calls[-1]["id"]

        tool_msg = {  # 构造一个 ToolMessage 表示交接成功
            "role": "tool",
            "content": "成功转移",
            "tool_call_id": tool_call_id,
        }
        # 核心：返回 Command 对象
        return Command(
            goto="multiplication_expert",  # 目的地：跳转到乘法专家节点
            update={"messages": [ai_msg, tool_msg]}  # 有效载荷：更新消息历史
        )
    # 如果不需要交接，就正常返回对消息的更新
    return {"messages": [ai_msg]}


def multiplication_expert(state: MessagesState, ) -> Command[Literal["addition_expert", "__end__"]]:
    system_prompt = (
        "您是乘法专家，您可以向加法专家寻求加法方面的帮助。 "
        "交接之前务必做好自己的那部分计算。"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_addition_expert]).invoke(messages)
    print(f"multiplication_expert:{ai_msg}")
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "成功转移",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="addition_expert", update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}


builder = StateGraph(MessagesState)
builder.add_node("addition_expert", addition_expert)
builder.add_node("multiplication_expert", multiplication_expert)
builder.add_edge(START, "addition_expert")
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='../../../../assets/Command交接.png')

from langchain_core.messages import convert_to_messages


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"从子图更新 {graph_id}:")
        print("\n")

    for node_name, node_update in update.items():
        print(f"从node更新 {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")


for chunk in graph.stream(
        {"messages": [("user", "(3 + 5) * 12 = ?")]},
):
    pretty_print_messages(chunk)
