'''
第一种方式:父图和子图共享模式键
'''
from langgraph.graph import START, StateGraph
from typing import TypedDict

# class SubgraphState(TypedDict):
#     foo: str  #这个键是与父图形状态共享的
#     bar: str
#
# def subgraph_node_1(state: SubgraphState):
#     return {"bar": "bar"}
#
# def subgraph_node_2(state: SubgraphState):
#     # 这个节点使用了一个只在子图中可用的状态键(“bar”)
#     # 并且同时发送共享状态键的更新('foo')
#     return {"foo": state["foo"] + state["bar"]}
#
#
# subgraph_builder = StateGraph(SubgraphState)
# subgraph_builder.add_node(subgraph_node_1)
# subgraph_builder.add_node(subgraph_node_2)
# subgraph_builder.add_edge(START, "subgraph_node_1")
# subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
#
# subgraph = subgraph_builder.compile()
#
# class ParentState(TypedDict):
#     foo: str
#
# def node_1(state: ParentState):
#     return {"foo": "hi! " + state["foo"]}
#
# builder = StateGraph(ParentState)
# builder.add_node("node_1", node_1)
# builder.add_node("node_2", subgraph)
# builder.add_edge(START, "node_1")
# builder.add_edge("node_1", "node_2")
# graph = builder.compile()
#
# for chunk in graph.stream({"foo": "foo"}):
#     print(chunk)
#
# print('##########################')
#
# for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
#     print(chunk)


'''
第二种方式:父图和子图具有不同的模式
'''
from langgraph.graph import START, StateGraph
from typing import TypedDict
class SubgraphState(TypedDict):
    # 这些键都不与父图形状态共享
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

"""调用子图的"""
def node_2(state: ParentState):
    # 将状态转换到子图状态
    response = subgraph.invoke({"bar": state["foo"]})
    # 将响应转换回父状态
    return {"foo": response["bar"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
    print(chunk)