from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from src.langchaindemo.model import getModel


class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        print(f"即将调用模型：{len(state['messages'])}个消息")
        return None

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        print(f"模型返回消息：{state['messages'][-1].content}")
        return None


model = getModel()
agent = create_agent(
    model=model,
    middleware=[LoggingMiddleware()],
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]}
)
print(result)
