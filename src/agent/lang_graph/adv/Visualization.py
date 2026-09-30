import random
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


class MyNode:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state: State):
        return {"messages": [("assistant", f"Called node {self.name}")]}


def route(state) -> Literal["entry_node", "__end__"]:
    if len(state["messages"]) > 10:
        return "__end__"
    return "entry_node"


def add_fractal_nodes(builder, current_node, level, max_level):
    if level > max_level:
        return

    # Number of nodes to create at this level
    num_nodes = random.randint(1, 3)  # Adjust randomness as needed
    for i in range(num_nodes):
        nm = ["A", "B", "C"][i]
        node_name = f"node_{current_node}_{nm}"
        builder.add_node(node_name, MyNode(node_name))
        builder.add_edge(current_node, node_name)

        # Recursively add more nodes
        r = random.random()
        if r > 0.2 and level + 1 < max_level:
            add_fractal_nodes(builder, node_name, level + 1, max_level)
        elif r > 0.05:
            builder.add_conditional_edges(node_name, route, node_name)
        else:
            # End
            builder.add_edge(node_name, "__end__")


def build_fractal_graph(max_level: int):
    builder = StateGraph(State)
    entry_point = "entry_node"
    builder.add_node(entry_point, MyNode(entry_point))
    builder.add_edge(START, entry_point)

    add_fractal_nodes(builder, entry_point, 1, max_level)

    # Optional: set a finish point if required
    builder.add_edge(entry_point, END)  # or any specific node

    return builder.compile()


app = build_fractal_graph(3)

from IPython.display import Image, display

"""可视化图"""
# 方式一：需要单独安装graphviz
# 方法：从官网下载
# 访问 Graphviz官网 https://graphviz.org/download/
# 下载Windows版本的安装包
# 运行安装程序并按照提示完成安装
# 将Graphviz的bin目录添加到系统PATH环境变量中（通常是C:\Program Files\Graphviz\bin）
# 并且请用try块包裹下面的语句，因为执行会报错，但是不影响图片的生成
try:
    display(Image(app.get_graph().draw_png(output_file_path='./可视化图1.png')))
except Exception as e:
    print(e)
# 方式二：不需要额外安装软件，但是访问网址mermaid.ink非常容易失败，开启科学上网比较容易成功
display(Image(app.get_graph().draw_mermaid_png(output_file_path='./可视化图2.png')))
