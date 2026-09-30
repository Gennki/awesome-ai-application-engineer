from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    age: int | None

# 标志图的开始运行
def show_begin_node(state: FormState):
    print(f"=====>show_begin_node :{state}")

# 标志图的结束运行
def show_end_node(state: FormState):
    print(f"show_end_node=====> :{state}")

def get_age_node(state: FormState):
    prompt = "你几岁了?"
    print(f"get_age_node begin :{prompt},state:{state}")
    inner_count = 0
    while True:
        inner_count = inner_count+1
        print(f"get_age_node 内部循环次数：{inner_count}")
        answer = interrupt(prompt)
        print(f"human answer :{answer}")
        if isinstance(answer, int) and answer > 0:
            # return {"age": answer}
            break
        prompt = f"'{answer}' 不是有效的年龄。请输入一个正数."
        print(f"------while_loop end ：inner_count:{inner_count},state:{state},{prompt}-------")
    print(f"=======get_age_node end：inner_count:{inner_count}=======")
    return {"age": answer}

builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_node("show_begin_node", show_begin_node)
builder.add_node("show_end_node", show_end_node)
builder.add_edge(START, "show_begin_node")
builder.add_edge("show_begin_node", "collect_age")
builder.add_edge("collect_age", "show_end_node")
builder.add_edge("show_end_node", END)
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "form-1"}}
first = graph.invoke({"age": None }, config=config)
print(f'第一次中断：{first["__interrupt__"]}')  # -> [Interrupt(value='What is your age?', ...)]
print(f'Command模拟人类输入"thirty"')
retry = graph.invoke(Command(resume="thirty"), config=config)
print(f'第二次中断：{retry["__interrupt__"]}')  # -> [Interrupt(value="'thirty' is not a valid age...", ...)]
print(f'Command第二次模拟人类输入"三十"')
third = graph.invoke(Command(resume="三十"), config=config)
print(f'第三次中断：{third["__interrupt__"]}')  # -> [Interrupt(value="'三十' is not a valid age...", ...)]
print(f'Command第三次模拟人类输入"30岁"')
fourth = graph.invoke(Command(resume="30岁"), config=config)
print(f'第四次中断：{fourth["__interrupt__"]}')  # -> [Interrupt(value="'30岁' is not a valid age...", ...)]
print(f'Command第四次模拟人类输入"30"')
final = graph.invoke(Command(resume=30), config=config)
print(f'final结果：{final["age"]}') # -> 30

"""
Command和interrupt()协同工作原理详解
整体工作机制
在LangGraph中，interrupt()和Command(resume=...)的协同工作遵循以下机制：
中断触发：当节点执行到interrupt()调用时，整个图的执行暂停，控制权返回给调用者
状态保存：LangGraph保存当前执行状态，包括未完成的节点信息
恢复执行：通过Command(resume=...)提供输入，图从保存的状态继续执行
重新执行：被中断的节点会从头开始重新执行，但interrupt()会直接返回resume的值而不是再次中断
LangGraph内部可能维护一个输入队列或栈，用于保存每个/每个Interrupt接收到的Command(resume=...)传递的值：
Input Queue: ["thirty"] -> ["thirty", "三十"] -> ["thirty", "三十", "30岁"] -> ["thirty", "三十", "30岁", "30"]
"""
