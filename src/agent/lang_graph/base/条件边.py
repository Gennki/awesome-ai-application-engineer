import operator
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    which: str


def a(state: State):
    print(f"Adding 'A' to {state['aggregate']}")
    return {"aggregate": ["A"]}


def b(state: State):
    print(f"Adding 'B' to {state['aggregate']}")
    return {"aggregate": ["B"]}


def c(state: State):
    print(f"Adding 'C' to {state['aggregate']}")
    return {"aggregate": ["C"]}


def d(state: State):
    print(f"Adding 'D' to {state['aggregate']}")
    return {"aggregate": ["D"]}


def e(state: State):
    print(f"Adding 'E' to {state['aggregate']}")
    return {"aggregate": ["E"]}


builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_node(e)

builder.add_edge(START, "a")

"""可自定义路由走单边"""
def route_bc_or_cd(state: State):
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]


intermediates = ["b", "c", "d"]
builder.add_conditional_edges(
    "a",
    route_bc_or_cd,
    path_map=intermediates,
)

for node in intermediates:
    print("intermediates", intermediates)
    builder.add_edge(node, "e")

builder.add_edge("e", END)
graph = builder.compile()
print(graph.invoke({"aggregate": [], "which": "cd"}))
