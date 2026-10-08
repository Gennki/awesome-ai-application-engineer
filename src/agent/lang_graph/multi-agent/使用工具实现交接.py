import json
from typing import Literal

from langchain_core.messages import ToolMessage, convert_to_messages
from langchain_core.tools import tool
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command

from src.langchaindemo.model import getModel

model = getModel()


def make_agent(model, tools, system_prompt=None):
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: MessagesState) -> Command[Literal["call_tools", "__end__"]]:
        messages = state["messages"]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        response = model_with_tools.invoke(messages)
        if len(response.tool_calls) > 0:
            # 确保工具调用参数格式正确
            corrected_tool_calls = []
            for tool_call in response.tool_calls:
                # 确保args是字典格式
                args = tool_call.get("args", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except:
                        args = {}

                corrected_tool_call = {
                    "name": tool_call["name"],
                    "args": args,
                    "id": tool_call.get("id", "")
                }
                corrected_tool_calls.append(corrected_tool_call)

            corrected_response = response.__class__(
                content=response.content,
                tool_calls=corrected_tool_calls,
                id=response.id
            )

            return Command(goto="call_tools", update={"messages": [corrected_response]})

        return {"messages": [response]}

    def call_tools(state: MessagesState) -> Command[Literal["call_model"]]:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            tool_ = tools_by_name[tool_call["name"]]

            # 确保参数是字典格式
            tool_args = tool_call.get("args", {})
            if isinstance(tool_args, str):
                # 如果是字符串，尝试解析为字典
                try:
                    tool_args = json.loads(tool_args) if tool_args else {}
                except:
                    tool_args = {}

            tool_input_fields = tool_.get_input_schema().model_json_schema()[
                "properties"
            ]

            if "state" in tool_input_fields:
                tool_call_updated = {**tool_call, "args": {**tool_args, "state": state}}
            else:
                tool_call_updated = {**tool_call, "args": tool_args}

            tool_response = tool_.invoke(tool_call_updated)
            if isinstance(tool_response, ToolMessage):
                results.append(Command(update={"messages": [tool_response]}))

            elif isinstance(tool_response, Command):
                results.append(tool_response)

        return results

    graph = StateGraph(MessagesState)
    graph.add_node(call_model)
    graph.add_node(call_tools)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_tools", "call_model")

    return graph.compile()


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"从子图更新 {graph_id}:")
        print("\n")

    # 检查update的类型并相应处理
    if isinstance(update, dict):
        for node_name, node_update in update.items():
            print(f"从node更新 {node_name}:")
            print("\n")

            # 检查node_update是否是字典且包含"messages"键
            if isinstance(node_update, dict) and "messages" in node_update:
                for m in convert_to_messages(node_update["messages"]):
                    m.pretty_print()
            # 如果node_update本身就是一个消息列表
            else:
                try:
                    for m in convert_to_messages(node_update):
                        m.pretty_print()
                except Exception as e:
                    print(f"无法打印消息: {node_update}")
            print("\n")
    else:
        print(f"更新内容: {update}")
        print("\n")


@tool
def add(a: int, b: int) -> int:
    """将两个数相加."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """将两个数相乘."""
    return a * b


# 测试单个agent
agent = make_agent(model, [add, multiply])
agent.get_graph().draw_mermaid_png(output_file_path='../../../../assets/使用工具实现交接-单个agent.png')
print("测试单个agent:")
for chunk in agent.stream({"messages": [("user", "(3 + 5) * 12")]}):
    pretty_print_messages(chunk)

# # 测试多agent交互
# def make_handoff_tool(*, agent_name: str):
#     """创建一个可以通过命令返回交接的工具"""
#     tool_name = f"transfer_to_{agent_name}"
#
#     @tool(tool_name)
#     def handoff_to_agent(
#         state: Annotated[dict, InjectedState],  # 注入当前图状态
#         tool_call_id: Annotated[str, InjectedToolCallId],): # 注入工具调用ID
#         """向其他代理寻求帮助."""
#         tool_message = {
#             "role": "tool",
#             "content": f"成功转移到 {agent_name}",
#             "name": tool_name,
#             "tool_call_id": tool_call_id,
#         }
#         # 核心：工具返回 Command 对象
#         return Command(
#             goto=agent_name, # 目的地：跳转到指定的 agent_name
#             graph=Command.PARENT, # 指示跳转发生在父图中（如果当前在子图中）
#             update={"messages": state["messages"] + [tool_message]}, # 有效载荷
#         )
#
#     return handoff_to_agent
#
# addition_expert = make_agent(
#     model,
#     [add, make_handoff_tool(agent_name="multiplication_expert")],
#     system_prompt="您是加法专家，您可以向乘法专家寻求乘法方面的帮助。",
# )
# multiplication_expert = make_agent(
#     model,
#     [multiply, make_handoff_tool(agent_name="addition_expert")],
#     system_prompt="您是乘法专家，您可以向加法专家寻求加法方面的帮助。",
# )
#
# builder = StateGraph(MessagesState)
# builder.add_node("addition_expert", addition_expert)
# builder.add_node("multiplication_expert", multiplication_expert)
# builder.add_edge(START, "addition_expert")
# graph = builder.compile()
# graph.get_graph().draw_mermaid_png(output_file_path='../../../../assets/使用工具实现交接-多个agent.png')
# print("\n测试多agent交互:")
# try:
#     for chunk in graph.stream(
#         {"messages": [("user", "(3 + 5) * 12")]}, subgraphs=True
#     ):
#         pretty_print_messages(chunk)
# except Exception as e:
#     print(f"执行出错: {e}")
