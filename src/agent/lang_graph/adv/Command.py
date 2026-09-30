import random
from typing import TypedDict, Literal

from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import Command


class State(TypedDict):
    foo: str


# Literal["node_b", "node_c"] 表示该函数返回的 Command 对象中的
# goto字段只能是 "node_b" 或 "node_c" 这两个字符串中的一个
def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["a", "b"])
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"
    print("goto:", goto)
    # Command中包含了更新图状态和路由到下一个节点
    return Command(
        update={"foo": value},
        goto=goto,
    )


def node_b(state: State):
    print("Called B")
    return {"foo": state["foo"] + "b"}


def node_c(state: State):
    print("Called C")
    return {"foo": state["foo"] + "c"}


builder = StateGraph(State)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
# 注意:上述代码中节点A、B、C之间没有边！

graph = builder.compile()
graph.invoke({"foo": ""})

"""
def lookup_user_info(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig):
    user_info = get_user_info(config.get("configurable", {}).get("user_id"))
    #判断用户类型，决定goto的取值
    goto="..."
    return Command(
        update={"user_info": user_info},
        goto=goto,
    )
"""
