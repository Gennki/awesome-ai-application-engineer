import sqlite3
from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    print(f"approval_node :{state}")
    # decision指interrupt被调用后，接收到的值(一般是人类输入)
    # interrupt的字典中的字段，可以显示在UI界面上
    decision = interrupt({
        "question": "批准此操作？",
        "details": state["action_details"],
    })
    # 恢复后路由到适当的节点
    return Command(goto="proceed" if decision else "cancel")

def proceed_node(state: ApprovalState):
    print(f"proceed_node :{state}")
    return {"status": "approved"}

def cancel_node(state: ApprovalState):
    print(f"cancel_node :{state}")
    return {"status": "rejected"}


builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
#approval并不会直接连接到"proceed"和"cancel"
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

# 使用持久检查点，这个必须
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

#这个也必须
config = {"configurable": {"thread_id": "approval-123"}}
initial = graph.invoke(
    {"action_details": "删除文件？", "status": "pending"},
    config=config,
)
print(initial["__interrupt__"])  # -> [Interrupt(value={'question': ..., 'details': ...})]

# 直接通过代码，继续做决定：True 继续，False 取消，继续图的执行要使用Command
resumed = graph.invoke(Command(resume=False), config=config)
print(resumed["status"])  # -> "approved"