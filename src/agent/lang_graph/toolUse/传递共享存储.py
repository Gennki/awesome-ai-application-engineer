from typing import Annotated, List, Tuple

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedStore, ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.langchaindemo.model import getModel

doc_store = InMemoryStore()

# 为两个不同的用户（ID ”1“ 和 ”2“）分别存储了一条关于FooBar公司的文档
namespace = ("documents", "1")  # user ID
doc_store.put(
    namespace, "doc_0", {"doc": "FooBar公司刚刚筹集了10亿美元！"}
)
namespace = ("documents", "2",)  # user ID
doc_store.put(
    namespace, "doc_1", {"doc": "FooBar公司成立于2019年"}
)


@tool
def get_context(
        question: str,
        config: RunnableConfig,  # 参数1：注入运行时配置
        store: Annotated[BaseStore, InjectedStore()]  # 参数2：注入文档存储
) -> Tuple[str, List[Document]]:
    """获取回答问题的相关背景"""
    # 从运行时配置中获取 user_id
    user_id = config.get("configurable", {}).get("user_id")
    # 从注入的store中根据user_id搜索问的那个
    docs = [item.value["doc"] for item in store.search(("documents", user_id))]
    return "\n\n".join(doc for doc in docs)


print(get_context.tool_call_schema.model_json_schema())

# 创建ReAct Agent 图并传递存储
tools = [get_context]
model = getModel()
# ToolNode会处理 InjectedStore 和 RunnableConfig 的注入
tool_node = ToolNode(tools)

checkpointer = MemorySaver()
# 注意：我们需要将我们的存储传递给“create_agent”，以确保我们的graph知道它
graph = create_agent(model, tools, checkpointer=checkpointer, store=doc_store)

# 第一次调用(user_id="1")
messages = [{"type": "user", "content": "关于FooBar有什么最新消息"}]
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# 第二次调用(user_id="2")
messages = [{"type": "user", "content": "关于FooBar有什么最新消息"}]
config = {"configurable": {"thread_id": "2", "user_id": "2"}}
for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
