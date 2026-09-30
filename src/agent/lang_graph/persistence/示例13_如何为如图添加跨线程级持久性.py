import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.langchaindemo.model import getEmbedding, getModel

# 初始化内存存储系统，配置嵌入模型和维度
# 使用它是因为已经内置了检索功能，不用自行实现
in_memory_store = InMemoryStore(
    index={
        "embed": getEmbedding(),
        "dims": 1024,  # 对应qwen3-embedding:0.6b模型的输出维度
    }
)

model = getModel()


# 定义核心处理函数
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # 从配置中获取用户ID，创建专属命名空间
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # 在存储中搜索与当前对话相关的记忆
    memories = store.search(namespace, query=str(state["messages"][-1].content))

    # 将记忆数据转换为字符串格式
    info = "\n".join([d.value["data"] for d in memories])
    print("info:", info, "#####")
    # 构建系统提示，包含用户记忆信息
    system_msg = f"你是一个与用户交流的好助手。用户信息：{info}"

    # 检查是否需要存储新记忆
    last_message = state["messages"][-1]
    print("last_message:", last_message)
    if "remember" in last_message.content.lower():
        # 生成并存储新的记忆（示例记忆内容）
        store.put(namespace, str(uuid.uuid4()), {"data": last_message.content})  # 使用UUID作为唯一键

    # 调用AI模型生成回复（结合系统提示词和对话历史）
    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


# 构建状态图工作流
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(
    checkpointer=MemorySaver(),  # 用于保存对话状态的检查点
    store=in_memory_store  # 使用之前配置的内存存储
)

graph.get_graph().draw_mermaid_png(output_file_path="../../../../assets/示例13.png")

# 测试场景1：存储记忆
config = {"configurable": {"thread_id": "1", "user_id": "1"}}  # 用户1的对话配置
input_message = {"role": "user", "content": "你好！remember：我的名字是张三"}
print("第一次对话（存储记忆）：")
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# 查看存储的记忆
print("\n存储的记忆内容：")
for memory in in_memory_store.search(("memories", "1",)):
    print(memory.value)
print("\n存储的记忆内容----------------")

# 测试场景2：读取记忆
config = {"configurable": {"thread_id": "3", "user_id": "1"}}
input_message = {"role": "user", "content": "我叫什么名字"}

print("\n第二次对话（读取记忆）：")
# 跨线程读取到记忆
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
