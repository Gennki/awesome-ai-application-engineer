from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph


class SubgraphState(TypedDict):
    foo: str
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

subgraph.get_graph().draw_mermaid_png(output_file_path="../../../../assets/示例12-子图.png")


class State(TypedDict):
    foo: str


def node_1(state: State):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")

# 使用内存检查器 MemorySaver 编译该图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path="../../../../assets/示例12.png")

# 验证持久性是否有效
config = {"configurable": {"thread_id": "1"}}
for _, chunk in graph.stream({"foo": "foo"}, config, subgraphs=True):
    print(chunk)

# 通过使用与调用图相同的配置来查看父图的状态
print(graph.get_state(config).values)
print("###")

# 检查父图的状态历史，找到在node_2（包含子图的节点）返回结果之前的状态快照
state_with_subgraph = [s for s in graph.get_state_history(config) if s.next == ("node_2",)][0]
print(state_with_subgraph)
print("###")

# 检索子图状态的配置
subgraph_config = state_with_subgraph.tasks[0].state
print(subgraph_config)
print("###")

print(graph.get_state(subgraph_config).values)
