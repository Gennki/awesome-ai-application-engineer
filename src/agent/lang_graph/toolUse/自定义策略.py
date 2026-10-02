import json
from typing import Literal

from langchain_core.messages import ToolMessage, AIMessage, RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from pydantic import Field, BaseModel

from src.langchaindemo.model import getModel


class HaikuRequest(BaseModel):
    topic: list[str] = Field(
        max_length=3,
        min_length=2
    )


@tool
def master_haiku_generator(request: HaikuRequest):
    """基于提供的主题生成俳句"""
    model = getModel()
    chain = model | StrOutputParser()
    topics = ",".join(request.topic)
    haiku = chain.invoke(f"Write a haiku about {topics}")
    return haiku


def call_tool(state: MessagesState):
    """
    获取传入状态中的最后一条消息
    遍历该消息中的所有工具调用
    尝试执行每个工具调用，成功则返回结果，失败则记录错误信息
    返回包含工具执行结果的消息列表
    """
    tools_by_name = {master_haiku_generator.name: master_haiku_generator}
    messages = state["messages"]
    last_message = messages[-1]
    output_messages = []
    for tool_call in last_message.tool_calls:
        try:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            output_messages.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            # 如果失败的处理
            output_messages.append(
                ToolMessage(
                    content="",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs={"error": e}
                )
            )
    return {"messages": output_messages}


model = getModel()
model_with_tools = model.bind_tools([master_haiku_generator])
better_model = getModel(model="deepseek-v4-flash")
better_model_with_tools = better_model.bind_tools([master_haiku_generator])


def should_continue(state: MessagesState):
    """
    检查最新消息是否有工具调用
    如果有工具调用，返回“tools”表示需要执行工具节点
    否则返回END表示流程结束
    """
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def should_fallback(
        state: MessagesState
) -> Literal["agent", "remove_failed_tool_call_attempt"]:
    """
    检查是否存在带有错误信息的工具消息
    如果存在失败的工具调用，返回“remove_failed_tool_call_attempt”节点
    否则返回“agent”节点继续正常流程
    """
    messages = state["messages"]
    """
    执行逻辑：在迭代过程中对每个元素逐一进行条件判断
    遍历消息列表：for msg in messages - 遍历状态中的所有消息
    类型检查：isinstance(msg,ToolMessage) - 确保消息是工具执行结果
    错误检查：msg.additional_kwargs.get("error") is not None - 检查消息中是否包含错误信息
    筛选结果：只保留同时满足上述两个条件的消息
    """
    failed_tool_messages = [
        msg for msg in messages if isinstance(msg, ToolMessage) and msg.additional_kwargs.get("error") is not None
    ]
    if failed_tool_messages:
        return "remove_failed_tool_call_attempt"
    return "agent"


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def remove_failed_tool_call_attempt(state: MessagesState):
    """
    查找最近的AIMessage消息
    删除从该消息开始的所有后续消息
    返回需要移除的消息列表
    """
    messages = state["messages"]
    # 从最近的AIMessage实例中删除所有消息
    last_ai_message_index = next(
        i
        for i, msg in reversed(list(enumerate(messages)))
        if isinstance(msg, AIMessage)
    )
    messages_to_remove = messages[last_ai_message_index:]
    return {"messages": [RemoveMessage(id=m.id) for m in messages_to_remove]}


# 如果模型工具调用失败，退回到更好的模型
def call_fallback_model(state: MessagesState):
    messages = state["messages"]
    response = better_model_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tool)
workflow.add_node("remove_failed_tool_call_attempt", remove_failed_tool_call_attempt)
workflow.add_node("fallback_agent", call_fallback_model)

"""
自定义错误处理策略的LangGraph工作流：

使用基础模型处理用户请求
如果需要工具调用，则执行相应工具
如果工具调用失败，则清除失败的尝试并切换到更强大的备用模型
继续执行直到完成
"""
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", should_fallback)
workflow.add_edge("remove_failed_tool_call_attempt", "fallback_agent")
workflow.add_edge("fallback_agent", "tools")
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="../../../../assets/自定义策略.png")

stream = app.stream(
    {"messages": [("human", "给我写一首关于水的绝妙俳句吧。主题关于水、河流、月光")]},
    {"recursion_limit": 10}
)

for chunk in stream:
    print(chunk)
