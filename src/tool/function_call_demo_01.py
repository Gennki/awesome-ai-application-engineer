# 定于工具函数
import platform
import subprocess
import webbrowser
from datetime import datetime
import gradio as gr

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import InMemorySaver

from src.langchaindemo.model import getModel


def get_current_time(input: str) -> str:
    """获取当前时间"""
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    result = f"当前时间{formatted_time}"
    print(result)
    return result


def recom_drink(input: str) -> str:
    """推荐附近的饮品店"""
    result = '''
   距离您500米以内有如下饮料店：\n
   1. 蜜雪冰城\n
   2. 茶颜悦色\n
   另外距离您200米内有惠民便利店，里面应该有矿泉水或其他饮品
   '''
    return result


def open_calc(input: str) -> str:
    """根据当前操作系统打开计算器。"""
    system = platform.system()
    commands = {
        "Windows": ["calc.exe"],
        "Darwin": ["open", "-a", "Calculator"],
    }
    command = commands.get(system)
    if command is None:
        return f"打开计算器失败：暂不支持当前操作系统 {system}"

    try:
        subprocess.Popen(command)
        return "计算器已打开"
    except OSError as error:
        return f"打开计算器失败：{error}"


def open_browser(url: str) -> str:
    """打开浏览器访问指定网址"""
    try:
        webbrowser.open(url)
        return f"已打开浏览器访问{url}"
    except Exception as e:
        return f"打开 浏览器失败：{str(e)}"


# 创建LangChain工具列表
# 通过@tool以外的第二种方式
tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        # 函数功能描述，给大模型看的说明书
        description="当你想知道现在的时间时可以使用"
    ),
    Tool(
        name="recom_drink",
        func=recom_drink,
        description="获取附近饮品店时调用"
    ),
    Tool(
        name="open_calc",
        func=open_calc,
        description="打开本地计算器"
    ),
    Tool(
        name="open_browser",
        func=open_browser,
        description="打开本地计算机上的网页浏览器，并接受网站的url作为参数"
    ),
]

model = getModel()
system_prompt = "你是一个人工智能助手，擅长帮助用户解决各种问题。"
# 创建短期记忆实例
memory = InMemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory,
)


# resp = agent.invoke(
#     {"messages": [{"role": "user", "content": "打开计算器"}]},
#     # 配置会话标识，用于区分不同用户
#     config={"configurable": {"thread_id": "user_1"}}
# )
# print(resp)

# 与前端交互处理LLM响应
def process_llm_response(query, show_history):
    show_history = show_history or []

    if not query:
        yield show_history, ""
        return

    user_message = {"role": "user", "content": query}

    try:
        yield show_history + [
            user_message,
            {"role": "assistant", "content": "正在查询大模型..."},
        ], ""

        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": "user_1"}},
        )
        print(f"LLM输出：{response}")

        if isinstance(response, dict) and response.get("messages"):
            last_message = response["messages"][-1]
            response_text = last_message.content
        else:
            response_text = str(response)
            print(f"警告：响应格式异常：{response_text}")

        yield show_history + [
            user_message,
            {"role": "assistant", "content": response_text},
        ], ""

    except Exception as error:
        print(f"Error：{error}")
        yield show_history + [
            user_message,
            {"role": "assistant", "content": "AI助手出错，请重试或者检查"},
        ], ""


# 前端界面展示
with gr.Blocks(title="大模型中Function Call演示") as demo:
    gr.HTML('<center><h1>欢迎来到Function Call演示</h1></center>')
    with gr.Row():
        with gr.Column(scale=10):
            chatbot = gr.Chatbot(height=320)

    with gr.Row():
        msg = gr.Textbox("输入", placeholder="您想了解什么呢？")

    with gr.Row():
        examples = gr.Examples(
            examples=[
                "请问如何做红烧肉？",
                "料酒可以换成白酒吗？",
                "帮我打开计算器",
                "现在几点了？",
                "帮我访问百度",
                "我渴了"
            ],
            inputs=[msg]
        )

    clear = gr.ClearButton([chatbot, msg])
    msg.submit(process_llm_response, [msg, chatbot], [chatbot, msg])

if __name__ == '__main__':
    demo.launch(server_port=7778)
