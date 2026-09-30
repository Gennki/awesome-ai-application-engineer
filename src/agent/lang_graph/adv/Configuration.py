import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv  # 用于加载环境变量

from src.langchaindemo.model import getModel

load_dotenv()  # 加载.env文件中的环境变量

glm_llm = getModel(model="z-ai/glm-5.3-flash")
deepseek_llm = getModel(model="deepseek/deepseek-v4.1-flash")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


from langchain_core.runnables.config import RunnableConfig

models = {
    "deepseek": deepseek_llm,
    "glm": glm_llm,
}


def _call_model(state: AgentState, config: RunnableConfig):
    # print('###',config["configurable"])

    model_name = config["configurable"].get("model", "glm")
    print('model_name：', model_name)
    model = models[model_name]
    response = model.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

config = {"configurable": {"model": "deepseek"}}
print(graph.invoke({"messages": [HumanMessage(content="你是谁？")]}, config=config))
config = {"configurable": {"model": "glm"}}
print(graph.invoke({"messages": [HumanMessage(content="你是谁？")]}, config=config))
