import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import HumanMessage, AnyMessage, AIMessage, SystemMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from src.langchaindemo.model import getModel

model = getModel()


# ================== 类型定义 ==================
class ChatState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]  # 自动合并消息的历史记录


# ================== 节点函数优化 ==================
def handle_user_input(state: ChatState):
    """处理用户输入节点"""
    try:
        print(f"handle_user_input_node state:{state}")
        user_input = input("\n用户输入（输入'退出'结束）: ").strip()
        if user_input.lower() == "退出":
            print("正在结束对话...")
            return END
        # # 保留历史记录并追加新消息
        # return {"messages": state["messages"] + [HumanMessage(content=user_input)]}
        return {"messages": [HumanMessage(content=user_input)]}
    except KeyboardInterrupt:
        return END


def generate_ai_response(state: ChatState):
    """生成AI响应节点（增加错误处理）"""
    try:
        print(f"generate_ai_response_node state:{state}")
        # 使用最近6条消息保持上下文连贯性
        recent_history = state["messages"][-6:]
        response = model.invoke(recent_history)
        print("模型答复：", response)
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"系统暂时无法响应，请稍后再试（错误代码：{str(e)[:30]})"
        return {"messages": [AIMessage(content=error_msg)]}


# ================== 对话图构建 ==================
builder = StateGraph(ChatState)
# 节点注册
builder.add_node("user_input", handle_user_input)
builder.add_node("ai_response", generate_ai_response)
# 流程设计
builder.set_entry_point("user_input")
builder.add_edge("user_input", "ai_response")
builder.add_edge("ai_response", END)
# 编译对话图
conversation = builder.compile()
# ================== 运行逻辑优化 ==================
if __name__ == "__main__":
    # 初始化带时间戳的系统提示
    system_prompt = f""" 你是一个专业级中文智能助手！"""
    state = ChatState(messages=[SystemMessage(content=system_prompt)])

    print("===== 智能对话系统已启动 =====")
    print("输入'退出'可随时结束对话\n")
    """
    1、每次调用 conversation.invoke(state) 都是一次独立的调用，
    虽然图流程定义了 END，但图本身不会阻止下一次调用。
    2、状态 state 会持续更新： 每次执行完 conversation.invoke(state) 后，
    返回的 result 会更新 state，从而保留了对话历史。
    当下次调用 invoke 时，传入的是更新后的 state，因此对话可以继续。
    3、END 只是图流程的结束点，不是程序的终止点：
    END 只是告诉 LangGraph 当前流程已结束，但并不会终止外部的 while 循环。
    """
    while True:
        try:
            # 执行对话流程，invoke后图执行一次完成
            result = conversation.invoke(state)
            print("图执行一次后：", result)
            if result is None or "messages" not in result:
                break
                # 提取最新交互记录
            new_messages = result["messages"][len(state["messages"]):]
            # 打印AI响应
            for msg in new_messages:
                if isinstance(msg, AIMessage):
                    print(f"\n【AI响应】\n{msg.content}\n")
                    # 更新对话状态
            state = result
            # 检查退出条件
            if any(isinstance(m, HumanMessage) and m.content.lower() == "退出" for m in state["messages"]):
                break
        except Exception as e:
            print(f"\n系统异常：{str(e)}")
            break
            # 对话结束处理
    print("\n===== 对话已结束 =====")
    print("\n【完整对话记录】")
    for i, msg in enumerate(state["messages"][1:]):  # 跳过系统提示
        prefix = "用户：" if isinstance(msg, HumanMessage) else "AI："
        print(f"{i + 1}. {prefix}{msg.content}")
