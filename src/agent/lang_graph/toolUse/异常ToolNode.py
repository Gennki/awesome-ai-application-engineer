from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from src.langchaindemo.model import getModel


@tool
def get_weather(location: str):
    """获取当前天气."""
    print('location：', location)
    if location == "SH":
        raise ValueError("输入查询必须是专有名词")
    elif location == "上海":
        return "气温23度，有雾."
    else:
        raise ValueError("无效输入.")


# tool_node = ToolNode([get_weather])
tool_node = ToolNode([get_weather], handle_tool_errors=True)

model_with_tools = getModel().bind_tools([get_weather])


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path='../../../../assets/异常ToolNode.png')

response = app.invoke({"messages": [("human", "长沙的天气怎么样?")]})

for message in response["messages"]:
    string_representation = f"{message.type.upper()}: {message.content}\n"
    print('string_representation：', string_representation)
