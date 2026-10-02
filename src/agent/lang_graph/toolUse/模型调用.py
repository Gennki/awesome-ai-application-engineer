from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from src.langchaindemo.model import getModel


@tool
def get_weather(location: str):
    """获取当前天气"""
    if location.lower() in ["SH", "上海"]:
        return "气温23度，有雾。"
    else:
        return "气温30度，阳光明媚。"


@tool
def get_coolest_cities():
    """获得最凉快城市列表"""
    return "青岛，上海"


tools = [get_weather, get_coolest_cities]
tools_node = ToolNode(tools)

model_with_tools = getModel().bind_tools(tools)

print(model_with_tools.invoke("上海的天气怎么样？"))
