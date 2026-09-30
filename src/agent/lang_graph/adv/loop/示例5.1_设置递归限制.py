import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

# Define nodes
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='../../../../../assets/示例5.1.png')


try:
    # 设置递归限制：可以1、自行判定；2，使用recursion_limit
    result = graph.invoke({"aggregate": []})
    # result = graph.invoke({"aggregate": []}, {"recursion_limit": 4})
    print(result)
except GraphRecursionError:
    print("Recursion Error")
