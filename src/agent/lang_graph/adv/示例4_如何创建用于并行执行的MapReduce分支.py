import operator
import re
from typing import Annotated

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph, START
from langgraph.types import Send
from typing_extensions import TypedDict

from src.langchaindemo.model import getModel

load_dotenv()

"""
由一个主题生成三个子主题，放到一个列表中，每个子主题并行生成一个笑话，最后从三个笑话中选择一个最好笑的
"""

# 增强的提示词模板
subjects_prompt = """作为主题专家，请为{topic}生成1-3个最相关的子主题，以逗号分隔。只返回子主题列表。"""
joke_prompt = """作为专业喜剧编剧，请创作一个关于{subject}的高质量笑话。要求：
1. 简洁幽默
2. 适合大众
3. 长度不超过2句话"""
best_joke_prompt = """作为幽默感评委，请从以下关于{topic}的笑话中选出最佳作品。评判标准：
1. 创意性(40%)
2. 幽默感(40%)
3. 语言表达(20%)

{jokes}

请只返回最佳笑话的编号(从0开始)。"""

model = getModel()
# 状态定义
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str
    best_joke_id: int

class JokeState(TypedDict):
    subject: str

def generate_topics(state: OverallState):
    """生成相关子主题"""
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.invoke(prompt)
    subjects = [
        s.strip()
        for s in re.split(r"[,，]+", response.content)
        if s.strip()
    ]
    return {"subjects": subjects[:3]}


def generate_joke(state: JokeState):
    """为每个子主题生成笑话"""
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.invoke(prompt)
    return {"jokes": [response.content]}


def continue_to_jokes(state: OverallState):
    """路由决定下一步，并行的执行"""
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


def best_joke(state: OverallState):
    """选择最佳笑话"""
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    try:
        response = model.invoke(prompt)
        # 尝试从响应中提取数字
        best_id = 0
        if response.content.isdigit():
            best_id = int(response.content)
        elif "0" in response.content or "1" in response.content or "2" in response.content:
            # 尝试从文本中提取数字
            for word in response.content.split():
                if word.isdigit():
                    best_id = int(word)
                    break

        # 确保ID在有效范围内
        best_id = max(0, min(best_id, len(state["jokes"])-1))
        best_joke = state["jokes"][best_id]

        return {
            "best_selected_joke": best_joke,
            "best_joke_id": best_id
        }
    except Exception as e:
        print(f"选择最佳笑话时出错: {e}")
        return {"best_selected_joke": state["jokes"][0]}  # 默认返回第一个



graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)

graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)
app = graph.compile()


app.get_graph().draw_mermaid_png(output_file_path='../../../../assets/示例4.png')
# 测试运行
if __name__ == "__main__":
    inputs = {"topic": "程序员"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"{key}: {value}")
        print("---")
