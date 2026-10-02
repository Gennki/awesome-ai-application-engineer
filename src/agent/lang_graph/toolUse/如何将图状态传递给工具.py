from typing import List, Annotated

from langchain.agents import AgentState, create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, InjectedState

from src.langchaindemo.model import getModel


class State(AgentState):
    docs: List[str]


@tool
# def get_context(question: str, state: State):
def get_context(question: str, state: Annotated[dict, InjectedState]):
    """获取回答问题的相关背景"""
    print('question', question)
    print('state', state)
    return "\n\n".join(doc for doc in state['docs'])


# get_input_schema可以理解为函数本身的参数情况显示
print('#', get_context.get_input_schema().model_json_schema())
# tool_call_schema可以理解为告诉大模型的工具调用参数
print('##', get_context.tool_call_schema.model_json_schema())

model = getModel()
tools = [get_context]
tool_node = ToolNode(tools)
checkpointer = MemorySaver()

graph = create_agent(model, tools, state_schema=State, checkpointer=checkpointer)

docs = [
    "FooBar公司刚刚筹集了10亿美元",
    "FoBar公司成立于2019年"
]

inputs = {
    "messages": [{"type": "user", "content": "关于FooBar有什么最新消息"}],
    "docs": docs
}

config = {"configurable": {"thread_id": "1"}}
for chunk in graph.stream(inputs, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
