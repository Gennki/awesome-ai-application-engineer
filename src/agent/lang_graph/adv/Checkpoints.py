from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

workflow = StateGraph(State)

workflow.add_node(node_a)
workflow.add_node(node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
print(graph.invoke({"foo": ""}, config))
#获取最新的状态快照
print("最新状态快照:",graph.get_state(config))
#获取状态快照的历史
print("状态快照历史:",list(graph.get_state_history(config)))
print()

# 获取特定检查点id的状态快照
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1f014733-8709-65ca-8001-52d231f87611"}}
print("指定ID状态快照:",graph.get_state(config))