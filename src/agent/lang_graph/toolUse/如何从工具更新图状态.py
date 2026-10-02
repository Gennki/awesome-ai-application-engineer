from typing import Annotated, Any

from langchain.agents import create_agent, AgentState
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.langchaindemo.model import getModel

USER_INFO = [
    {"user_id": "1", "name": "Bob Dylan", "location": "New York, NY"},
    {"user_id": "2", "name": "Taylor Swift", "location": "Beverly Hills, CA"},
]
USER_ID_TO_USER_INFO = {info["user_id"]: info for info in USER_INFO}


class State(AgentState):
    user_info: dict[str, Any]


@tool
def lookup_user_info(
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig
):
    """使用此功能查找用户信息，以便更好地帮助他们解答问题"""
    user_id = config.get("configurable", {}).get("user_id")
    if user_id is None:
        raise ValueError("请提供用户ID")
    if user_id not in USER_ID_TO_USER_INFO:
        raise ValueError(f"用户 '{user_id}' 没找到")
    user_info = USER_ID_TO_USER_INFO[user_id]
    return Command(
        update={
            "user_info": user_info,
            "messages": [
                ToolMessage(
                    "成功查询用户信息", tool_call_id=tool_call_id
                ),
                SystemMessage(content=f"用户信息：{user_info}")
            ]
        }
    )


model = getModel()
static_system_prompt = "你是一个乐于助人的AI助手。如果需要用户信息，你应该调用 lookup_user_info 工具来获取。"

agent = create_agent(
    model,
    [lookup_user_info],
    state_schema=State,
    system_prompt=static_system_prompt
)

for chunk in agent.stream(
        {"messages": [("user", "你好，这个周末我应该做什么？")]},
        {"configurable": {"user_id": "1"}}
):
    print(chunk)
    print("\n")

for chunk in agent.stream(
        {"messages": [("user", "你好，这个周末我应该做什么？")]},
        {"configurable": {"user_id": "2"}}
):
    print(chunk)
    print("\n")
