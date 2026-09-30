from datetime import datetime
from typing import TypedDict, Annotated

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph

from src.langchaindemo.model import getModel

model = getModel(temperature=0.7)


@tool
def get_current_time():
    """返回当前的日期和时间。当用户问'现在几点'、'今天日期'时调用此工具。"""
    print("=====enter in======")
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str):
    """计算数学表达式。输入应该是一个字符串形式的数学表达式，如'23*45+100'"""
    try:
        print("执行自定义算术运算")
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression}={result}"
    except Exception as e:
        return f"计算失败：{str(e)}"


agent = create_agent(
    model=model,
    tools=[get_current_time, calculator],
    system_prompt="你是一个友好的AI助手，用简洁易懂的语言回答用户问题。"
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: AgentState):
    response = agent.invoke({"messages": state["messages"]})
    return {"messages": [response["messages"][-1]]}


graph = StateGraph(AgentState)

graph.add_node("agent", call_model)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

checkpointer = InMemorySaver()
graph = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "user_123"}}
result1 = graph.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    config=config
)
print("Agent:", result1["messages"][-1].content)

result2 = graph.invoke(
    {"messages": [{"role": "user", "content": "现在几点"}]},
    config=config
)
print("Agent:", result2["messages"][-1].content)

result3 = graph.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字"}]},
    config=config
)
print("Agent:", result3["messages"][-1].content)

result4 = graph.invoke(
    {"messages": [{"role": "user", "content": "23乘50+300等于多少"}]},
    config=config
)
print("Agent:", result4["messages"][-1].content)
