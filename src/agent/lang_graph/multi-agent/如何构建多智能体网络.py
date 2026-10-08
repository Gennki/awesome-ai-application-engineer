from typing import Literal

from langchain_core.messages import convert_to_messages
from langchain_core.tools import tool
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command

from src.langchaindemo.model import getModel

model = getModel()


@tool
def transfer_to_travel_advisor():
    """向旅行顾问寻求帮助"""
    # 此工具不返回任何内容：我们只是使用它作为LLM发出需要移交给其他代理的信号
    return


@tool
def transfer_to_hotel_advisor():
    """向酒店顾问寻求帮助"""
    return


def travel_advisor(state: MessagesState) -> Command[Literal["hotel_advisor", "__end__"]]:
    system_prompt = (
        "您是一位可以推荐旅游目的地（例如国家、城市等）的综合旅游专家。"
        "如果您需要酒店推荐，请向'hotel_advisor'寻求帮助。"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_hotel_advisor]).invoke(messages)
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="hotel_advisor", update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}


def hotel_advisor(state: MessagesState) -> Command[Literal["travel_advisor", "__end__"]]:
    system_prompt = (
        "您是一位酒店专家。您的任务是承接 'travel_advisor' 的工作。"
        "**请仔细检查完整的对话历史记录。'travel_advisor' 在调用工具将任务移交给您之前，一定会在其回复中明确推荐一个旅游目的地（例如 '巴巴多斯' 或 '巴哈马'）。**"
        "您的任务是：1. 在历史记录中找到这个目的地。 2. 为这个已确定的目的地提供酒店推荐。"
        "**请不要评论“转移”这个过程本身，直接开始推荐酒店。**"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_travel_advisor]).invoke(messages)
    # 如果有工具调用，LLM 需要移交给另一个代理
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]

        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="travel_advisor", update={"messages": [ai_msg, tool_msg]})

    # 如果专家有答案，则直接返回给用户
    return {"messages": [ai_msg]}


builder = StateGraph(MessagesState)
builder.add_node("travel_advisor", travel_advisor)
builder.add_node("hotel_advisor", hotel_advisor)
builder.add_edge(START, "travel_advisor")
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="../../../../assets/构建多智能体网络.png")


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        # 跳过打印输出中的父图更新
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")

    for node_name, node_update in update.items():
        print(f"Update from node {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")


for chunk in graph.stream(
        {
            "messages": [
                ("user", "我想去加勒比海某个温暖的地方，选一个目的地，然后给我推荐酒店")
            ]
        }
):
    pretty_print_messages(chunk)
