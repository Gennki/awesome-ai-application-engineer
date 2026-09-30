"""Send到同质性节点"""
import operator
from typing import Annotated
from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.types import Send
from langgraph.graph import END, START


# 定义一个名为OverallState的TypedDict类
class OverallState(TypedDict):
    # subjects是一个字符串列表
    subjects: list[str]
    # jokes是一个带有operator.add注解的字符串列表
    jokes: Annotated[list[str], operator.add]

# 定义一个函数continue_to_jokes，接受一个OverallState类型的参数state
def continue_to_jokes(state: OverallState):
    # 返回一个Send对象的列表，每个对象包含一个"generate_joke"的node和传递给node的状态
    # subjects中有几个元素，就会有几个Send对象
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

# 创建一个StateGraph对象builder，传入OverallState类型
builder = StateGraph(OverallState)

# 添加一个名为"generate_joke"的节点，节点执行一个lambda函数，生成一个关于主题的笑话
builder.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subject']}"]})

# 添加一个条件边，从START节点到continue_to_jokes函数返回的节点
builder.add_conditional_edges(START, continue_to_jokes)

# 添加一条边，从"generate_joke"节点到END节点
builder.add_edge("generate_joke", END)

# 编译graph，生成最终的graph对象
graph = builder.compile()

# 调用graph对象，并传入包含两个主题的初始状态，结果是为每个主题生成一个笑话
result = graph.invoke({"subjects": ["cats", "dogs"]})
print(result)

# 将生成的图片保存到文件，这里因为要访问网络，可能会失败，请多试几次
# graph_png = graph.get_graph().draw_mermaid_png(output_file_path='./MR.png')
