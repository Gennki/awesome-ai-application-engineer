import re
import uuid
from typing import TypedDict, Annotated

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.constants import START
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langchaindemo.model import getEmbedding, getModel


def create_tool(company: str) -> dict:
    """为占位符工具创建架构"""
    # 删除工具名称中的非字母数字字符，并用下划线替代空格
    formatted_company = re.sub(r"[^\w\s]]", "", company).replace(" ", "_")

    def company_tool(year: int) -> str:
        # 返回公司和年份的静态收入信息的占位符函数
        return f"{company} had revenues of $100 in {year}."

    return StructuredTool.from_function(
        company_tool,
        name=formatted_company,
        description=f"Information about {company}",
    )


# 标准普尔 500 指数公司简略列表（用于演示）
s_and_p_500_companies = [
    "3M",
    "A.O. Smith",
    "Abbott",
    "Accenture",
    "Advanced Micro Devices",
    "Yum! Brands",
    "Zebra Technologies",
    "Zimmer Biomet",
    "Zoetis",
]

# 为每个公司创建一个工具，并将其存储在具有唯一 UUID 作为秘钥的注册表中
tool_registry = {
    str(uuid.uuid4()): create_tool(company) for company in s_and_p_500_companies
}
print(tool_registry)

tool_documents = [
    Document(
        page_content=tool.description,
        id=id,
        metadata={"tool_name": tool.name}
    )
    for id, tool in tool_registry.items()
]
print(tool_documents)

vector_store = InMemoryVectorStore(embedding=getEmbedding())
document_ids = vector_store.add_documents(tool_documents)


# 使用 TypedDict 定义状态结构。
# 它包含一个消息列表（由 add_messages 处理）
# 以及一个选定工具 ID 的列表。
class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]  # 存储选定工具的ID


builder = StateGraph(State)


def select_tools(state: State):
    last_user_message = state["messages"][-1]
    query = last_user_message.content  # 获取用户的最新问题
    # 在向量存储中搜索与用户问题最相关的工具描述
    tool_documents = vector_store.similarity_search(query)
    # 提取相关工具的ID，并更新到状态的selected_tools字段
    return {"selected_tools": [document.id for document in tool_documents]}


tools = list(tool_registry.values())  # 包含了所有工具，但agent会自动选择
print(tools)

model = getModel()


def agent(state: State):
    # 从状态中获取当前轮次被选中的工具ID
    selected_tool_ids = state["selected_tools"]
    # 根据ID从完整的工具注册表中获取实际的工具对象
    current_tools_for_llm = [tool_registry[id] for id in selected_tool_ids]
    # 将这些选中的工具绑定到LLM，LLM在这次调用中只会看到这些工具
    llm_with_selected_tools = model.bind_tools(current_tools_for_llm)
    # 调用LLM，只使用选中的工具
    return {"messages": [llm_with_selected_tools.invoke(state["messages"])]}


builder.add_node("agent", agent)
builder.add_node("select_tools", select_tools)

tool_node = ToolNode(tools=tools)
builder.add_node("tools", tool_node)

builder.add_conditional_edges("agent", tools_condition, path_map=["tools", "__end__"])
builder.add_edge("tools", "agent")
builder.add_edge("select_tools", "agent")
builder.add_edge(START, "select_tools")
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='../../../../assets/如何处理大量工具.png')

user_input = "你能给我一些关于 2022 年 AMD 的信息吗?"

result = graph.invoke({"messages": [("user", user_input)]})
print(result["selected_tools"])

for message in result["messages"]:
    message.pretty_print()
