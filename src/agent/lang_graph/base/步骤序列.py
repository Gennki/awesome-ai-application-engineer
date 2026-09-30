from typing import TypedDict

from langgraph.constants import START
from langgraph.graph import StateGraph


class State(TypedDict):
    value_1: str
    value_2: int


def step_1(state: State):
    print(state)
    return {"value_1": "a"}


def step_2(state: State):
    current_value_1 = state["value_1"]
    return {"value_1": f"{current_value_1} + b"}


def step_3(state: State):
    print(state)
    return {"value_2": 10}


graph_builder = (StateGraph(State)
                 .add_sequence([step_1, step_2, step_3]))
graph_builder.add_edge(START, "step_1")
graph = graph_builder.compile()

print(graph.invoke({"value_1": "c"}))
