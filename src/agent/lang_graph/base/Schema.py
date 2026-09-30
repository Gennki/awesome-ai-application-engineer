"""
注意：本程序运行无业务意义，仅展示知识点
"""
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# 定义输入Schema
class InputState(TypedDict):
    question: str


# 定义输出Schema
class OutputState(TypedDict):
    answer: str


# 结合输入和输出，定义总体模式
class OverallState(InputState, OutputState):
    pass


# 定义处理输入并生成答案的节点
def answer_node(state: InputState):
    return {"answer": "hello", "question": state["question"]}


builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node("answer_node", answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)
graph = builder.compile()

print(graph.invoke({"question": "hi"}))
# 可以运行，但是传入的"answer"不会起作用
print(graph.invoke({"question": "hi", "answer": "OK"}))
