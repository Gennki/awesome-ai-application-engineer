from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# 图的总体状态（这是跨节点共享的公共状态）
class OverallState(TypedDict):
    a: str


# node_1的输出 包含不属于整体状态的私有数据，将 Node1 -> Node2
class Node1Output(TypedDict):
    private_data: str


# node_2的输入 包含不属于整体状态的私有数据
class Node2Input(TypedDict):
    private_data: str


# 私有数据只在node_1和node_2之间共享
def node_1(state: OverallState) -> Node1Output:
    output = {"private_data": "set by node_1"}
    print(f"Entered node `node_1`:\n\tInput: {state}.\n\toutput: {output}")
    return output


# node2的输入 只请求node_1之后可用的私有数据
def node_2(state: Node2Input) -> OverallState:
    output = {"a": "set by node_2"}
    print(f"Entered node `node_2`:\n\tInput: {state}.\n\toutput: {output}")
    return output


# node3看不到在node_1和node_2之间流转的的私有数据）
def node_3(state: OverallState) -> OverallState:
    output = {"a": "set by node_3"}
    print(f"Entered node `node_3`:\n\tInput: {state}.\n\toutput: {output}")
    return output


builder = StateGraph(OverallState)

builder.add_node(node_1)  # Node_1是第一个节点
builder.add_node(node_2)  # Node_2是第二个节点，接受来自node_1的私有数据
builder.add_node(node_3)  # Node_3是第三个节点，没有看到私有数据

builder.add_edge(START, "node_1")  # 以node_1开始图
builder.add_edge("node_1", "node_2")  # 从node_1传递到node_2
builder.add_edge("node_2", "node_3")  # 从node_2传递到node_3（仅共享整体状态）
builder.add_edge("node_3", END)  # 结束node_3后的图

graph = builder.compile()

# 调用具有初始状态的图形
response = graph.invoke({"a": "set at start", })
print()
print(f"输出: {response}")
