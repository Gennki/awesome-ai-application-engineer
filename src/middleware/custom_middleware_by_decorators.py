from typing import Callable, Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model, wrap_model_call, ModelRequest, ModelResponse
from langgraph.runtime import Runtime

from src.langchaindemo.model import getModel


# 前置。在模型调用前，执行这个函数
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"即将调用模型：{len(state['messages'])}个消息")
    return None


# 环绕，在模型调用前后都执行这个函数
@wrap_model_call
def round_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    print(f"模型调用前置处理 request：request={request}")
    print(f"模型调用前置处理 handler：handler={handler}")

    result = handler(request)  # 调用模型

    print(f"模型调用后，模型返回结果：{result}")
    return result


model = getModel()
agent = create_agent(
    model=model,
    middleware=[log_before_model, round_model],
)
result = agent.invoke(
    {"messages": [{
        "role": "user",
        "content": "你好"
    }]}
)
print(result["messages"][-1].content)
