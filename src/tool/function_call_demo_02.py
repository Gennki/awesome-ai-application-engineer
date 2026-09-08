from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.checkpoint.memory import InMemorySaver

from src.langchaindemo.model import getModel

# 构建一个基于arxiv工具的论文查询智能体。实现根据论文编号查询论文信息的功能。
# arxiv是LangChain中用于检索学术论文的工具,使用load_tools()导入arxiv工具。
tools = load_tools(["arxiv"])

# 创建短期记忆实例
memory = InMemorySaver()

# 系统提示词
system_prompt = "你是一个专业的论文查询助手，使用arxiv工具为用户查询论文信息。回答需简洁准确。包含论文标题、作者、发表时间和核心摘要"

# 组装agent
agent = create_agent(
    model=getModel(),
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory
)

result = agent.invoke(
    input={"messages": [{"role": "user", "content": "请查询arxiv论文编号1605.08386的信息"}]},
    config={"configurable": {"thread_id": "user_1"}}
)
print(result["messages"][-1].content)
