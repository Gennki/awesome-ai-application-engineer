"""
自定义中间件实现脱敏
"""
import re
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from src.langchaindemo.model import getModel


# 自定义中间件，去除电话号码和邮箱信息
class DesensitizeDataMiddleware(AgentMiddleware):
    def __init__(self, patterns: list = None):
        super().__init__()
        self.patterns = patterns or [
            (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[EMAIL]'),
            (r'(\+86)?1[3-9]\d{9}', '[PHONE]')
        ]

    def _desensitize_text(self, text: str) -> str:
        if not text or '[EMAIL]' in text or '[PHONE]' in text:
            return text

        if '@' not in text and not re.search(r'(?:\+86)?1[3-9]\d{9}', text):
            return text

        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)
        return text

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        messages = state.get('messages', [])
        processed_count = 0
        for message in messages:
            if not hasattr(message, 'content') or not isinstance(message.content, str):
                continue
            sanitized = self._desensitize_text(message.content)
            if sanitized != message.content:
                message.content = sanitized
                processed_count += 1

        if processed_count:
            print(f"已脱敏 {processed_count} 条消息")
        return state

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return state


def main() -> None:
    agent = create_agent(
        model=getModel(),
        tools=load_tools(["arxiv"]),
        system_prompt=(
            "你是一个专业的论文查询助手，使用 arxiv 工具查询论文信息。"
            "回答包含标题、作者、发表时间和核心摘要。"
        ),
        checkpointer=InMemorySaver(),
        middleware=[DesensitizeDataMiddleware()],
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我的邮箱是test.user@example.com。请帮我查询论文1605.08386",
                }
            ]
        },
        config={"configurable": {"thread_id": "middleware_test_1"}},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
