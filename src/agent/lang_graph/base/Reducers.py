# 定义状态类型，使用TypedDict明确字段类型
from operator import add
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    # foo字段为整数类型，更新时直接替换
    foo: int
    # bar字段为字符串列表，使用Annotated标注合并策略为add（列表拼接）
    bar: Annotated[list[str], add]  # map高阶函数


graph_builder = StateGraph(State)


def example_node(state: State):
    print(state)
    print(state["bar"] + ["OK"])
    return {"bar": state["bar"] + ["OK"]}


graph_builder.add_node("example_node", example_node)
graph_builder.add_edge(START, "example_node")
graph_builder.add_edge("example_node", END)
graph = graph_builder.compile()
print(graph.invoke({"bar": ["My", "name"], "foo": 1}))
