from typing import Literal

from langchain_core.messages import RemoveMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph

from src.langchaindemo.model import getModel

memory = MemorySaver()
model = getModel()


class State(MessagesState):
    summary: str


def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earliear:{summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# 自定义用于确定是结束还是总结对话的逻辑
def should_continue(state: MessagesState) -> Literal["summarize_conversation", END]:
    """返回下一个要执行的节点"""
    messages = state["messages"]
    # 如果有六条以上的消息，那么我们将对对话进行总结
    if len(messages) > 6:
        return "summarize_conversation"
    return END


def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_messages = (
            f"这是迄今为止的对话摘要：{summary}\n\n"
            "通过考虑上述新消息来扩展摘要："
        )
    else:
        summary_messages = "创建上述对话的摘要"

    messages = state["messages"] + [HumanMessage(content=summary_messages)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    print("delete_messages:", delete_messages)
    return {"summary": response.content, "messages": delete_messages}


workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

app = workflow.compile(checkpointer=memory)
app.get_graph().draw_mermaid_png(output_file_path="../../../../assets/添加会话历史摘要.png")


def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])


config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="你好，我是张三")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="我叫什么名字")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="我喜欢曼联！")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="我喜欢他们总是赢球")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

values = app.get_state(config).values
print(values)

input_message = HumanMessage(content="我叫什么名字")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="你觉得我喜欢哪支英超球队？")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 当前线程状态
messages = app.get_state(config).values["messages"]
print(messages)

# RemoveMessage是一种特殊的消息类型
# MessagesState中的归约函数add_messages中会对RemoveMessage进行特殊处理
# id=REMOVE_ALL_MESSAGES 删除所有消息
app.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})
messages = app.get_state(config).values["messages"]
print("现在", messages)
