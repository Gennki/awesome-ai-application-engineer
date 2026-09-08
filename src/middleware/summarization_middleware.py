from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.langchaindemo.model import getModel

model = getModel()
memory = InMemorySaver()
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=memory,
    # 中间件列表，可以多个，多个顺序执行
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 80),
            keep=("messages", 1),
            # 可选，summary_prompt="可以自定义进行摘要的提示词..."
            summary_prompt="请讲一下对话历史进行简洁的摘要，保留关键信息：{messages}"
        )
    ],
    # 打印Agent执行的过程日志
    debug=True
)

print("\n模拟长对话场景...")
demo_messages = [
    "用户询问你是谁",
    "用户计算商品价格：数量10，单价25.5",
    "用户再次询问你能做什么？",
    "用户想要生成一个介绍江苏的文案，要求100字左右",
    "用户想要继续询问更多GPU产品信息",
    "用户要求计算2*10",
]
for i, message in enumerate(demo_messages):
    print(f"\n第{i}轮对话：{message}")
    # 循环调用Agent，模拟多轮对话
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": "test_summarization_middleware"}}
    )
