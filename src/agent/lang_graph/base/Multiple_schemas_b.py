from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class InputState(TypedDict):
    user_input: str


class OutputState(TypedDict):
    graph_output: str


class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str


class PrivateState(TypedDict):
    bar: str


def node_1(state: InputState) -> OverallState:
    # 写入OverallState
    print("node_1:", state)
    return {"foo": state["user_input"] + " name"}


def node_2(state: OverallState) -> PrivateState:
    # 从OverallState读取, 写入PrivateState
    print("node_2:", state)
    return {"bar": state["foo"] + " is"}


def node_3(state: PrivateState) -> OutputState:
    # 从PrivateState读取, 写入OutputState
    print("node_3:", state)
    return {"graph_output": state["bar"] + " YunFang"}


builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
# builder = StateGraph(OverallState)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()

print(graph.invoke({"user_input": "My"}, verbose=True))
