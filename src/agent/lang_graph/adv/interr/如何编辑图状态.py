from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display


class State(TypedDict):
    input: str

def step_1(state):
    print("---Step 1---")
    pass

def step_2(state):
    print("---Step 2---")
    pass

def step_3(state):
    print("---Step 3---")
    pass

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)


memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["step_2"])
graph.get_graph().draw_mermaid_png(output_file_path='../../../../../assets/如何编辑图状态.png')

initial_input = {"input": "hello world"}
thread = {"configurable": {"thread_id": "1"}}

# 运行graph，直到第一次中断
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

print("目前state!")
print(graph.get_state(thread).values)

graph.update_state(thread, {"input": "你好宇宙!"})

print("---\n---\n更新state!")
print(graph.get_state(thread).values)

# 继续执行
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
