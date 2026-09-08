"""
业务场景:电商客户反馈处理系统

需求描述:
某电商平台需要自动处理客户反馈，实现以下功能:
1. 情感分析:判断用户反馈的情感倾向
2. 问题分类:识别反馈的问题类型
3. 紧急程度评估:根据内容判断处理优先级
4. 生成回复草稿:根据分析结果生成初步回复
"""
import re
import time
from http.client import HTTPException

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from pydantic import BaseModel

from src.langchaindemo.model import getModel

model = getModel()


# 带重试的模型调用
def call_llm(prompt, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            print(f"调用LLM大模型，问题：{prompt}")
            response = model.invoke(prompt)
            print(f"调用LLM大模型，回复：{response.content}")
            return response.content
        except Exception as e:
            print(f"模型调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(retry_delay)
    return "模型服务暂不可用,请稍后再试."


# 提取订单ID
def extract_order_id(text: str) -> dict:
    # 正则提取
    match = re.search(r'(?<!\d)ORD\d{10}(?!\d)', text)
    if match:
        return {"order_id": match.group(0)}
    else:
        # 备选方案:大模型提取
        prompt = f"""
                        你是一个电脑订单处理专家,请从以下用户反馈中提取订单ID:
                        {text}

                        订单ID是"ORD"开头加固定10位数字组合.如果找不到订单ID,返回"NOT_FOUND".

                        请严格按JSON格式返回结果:{{"order_id":"提取结果"}}
                        """
        result = call_llm(prompt)
        # 尝试解析JSON
        try:
            return JsonOutputParser().parse(result)
        except Exception:
            return {"order_id": "NOT_FOUND"}


# 情感分类
def analyze_sentiment(text: str) -> dict:
    prompt = f"""
    请分析以下客户反馈的情感倾向:
    [{text}]
    
    要求:
    1. 判断情感类型:POSITIVE(积极)/NEUTRAL(中性)/NEGATIVE(消极)
    2. 评估置信度(0.0-1.0)
    3. 提取3个关键短语
    
    返回JSON格式:
    {{
        "sentiment":"情感类型",
        "confidence":置信度,
        "key_phrases":["短语1","短语2","短语3"]
    }}
    """
    try:
        result = call_llm(prompt)
        parser = JsonOutputParser()
        return parser.parse(result)
    except Exception as e:
        print(f"情感分析失败: {str(e)}")
        return {
            "sentiment": "NEUTRAL",
            "confidence": 0.7,
            "key_phrases": []
        }


# 问题分类
def classify_issue(text: str) -> dict:
    prompt = f"""
    作为电商客服专家,请对以下客户反馈进行分类：
    [{text}]
    
    分类选项：
    - 物流问题：配送延迟、物流损坏等
    - 产品质量：商品瑕疵、功能故障等
    - 客户服务：客服态度、响应速度等
    - 支付问题：扣款异常、退款延迟等
    - 退货退款：退货流程、退款金额等
    - 其他：无法归类的反馈
    
    要求：
    1. 选择最相关的1-2个分类
    2. 按相关性排序
    
    返回JSON格式：{{"categories":["分类1","分类2"]}}
    """
    try:
        result = call_llm(prompt)
        parser = JsonOutputParser()
        return parser.parse(result)
    except Exception as e:
        print(f"问题分类失败：: {str(e)}")
        return {"categories": "其他"}


# 紧急状态评估
def access_urgency(text: str) -> dict:
    prompt = f"""
    作为客服主管，请评估以下客户反馈的紧急程度：
    [{text}]
    
    评估标准：
    - HIGH(高)：包含"紧急"、"立刻"、"马上"或威胁投诉
    - MEDIUM(中)：表达强烈不满但无立即行动要求
    - LOW(低)：一般反馈或建议
    
    返回JSON格式：
    {{
        "urgency":"紧急级别",
        "sla_hours":响应时限(小时),
        "reason":"评估理由"
    }}
    """
    try:
        result = call_llm(prompt)
        parser = JsonOutputParser()
        return parser.parse(result)
    except Exception as e:
        print(f"紧急度评估失败: {str(e)}")
        return {
            "urgency": "MEDIUM",
            "sla_hours": 24,
            "reason": "评估失败"
        }


def generate_response(data: dict) -> dict:
    print(data)
    """使用大模型生成定制化回复"""
    prompt_template = """
    你是一名资深电商客服专家.请根据以下分析结果生成客户回复:

    ### 客户反馈原文:
    {feedback}

    ### 分析结果:
    - 订单ID:{order_id}
    - 情感倾向:{sentiment}(置信度:{confidence:.2f})
    - 问题类型:{categories}
    - 紧急程度:{urgency}(需在{sla_hours}小时内响应)
    {key_phrases_section}

    ### 回复要求:
    1. 根据情感倾向调整语气:
        - 积极反馈:表达感谢.适当赞美
        - 消极反馈:诚恳道歉.明确解决方案
    2. 包含订单ID和问题分类
    3. 明确说明处理时限和后续步骤
    4. 长度100-150字,使用自然口语
    5. 结尾询问是否还有其他问题
    6. 如果订单ID是“NOT_FOUND”，表示此订单不存在

    请直接输出回复内容,不需要额外说明.
    """
    key_phrases = data.get("key_phrases", [])
    if key_phrases:
        key_phrases_section = "- 关键要点:" + ",".join(key_phrases[:3])
    else:
        key_phrases_section = ""

    # 填充模板
    prompt = prompt_template.format(
        feedback=data["original_feedback"],
        order_id=data["order_id"],
        sentiment=data["sentiment"],
        confidence=data["confidence"],
        categories=data["categories"],
        urgency=data["urgency"],
        sla_hours=data["sla_hours"],
        key_phrases_section=key_phrases_section
    )

    try:
        response = call_llm(prompt)
        # 添加紧急标识
        if data["urgency"] == "HIGH":
            response = f"[紧急] {response}"

        return {
            "final_response": response,
            "assigned_team": data["categories"][0] if data["categories"] else "General",
            "result": data
        }

    except Exception as e:
        print(f"回复生成失败: {str(e)}")
        return {
            "final_response": "感谢您的反馈,我们的团队将尽快处理您的问题.",
            "assigned_team": "General"
        }


# 步骤1:基础信息提取
"""
返回的数据格式:
{
    "order_id":{
        "order_id":"ORD2026081416" # 或 "NOT_FOUND"
    },
    "original_feedback":"用户原始反馈文本"
}
"""
extract_chain = RunnableParallel(
    order_id=RunnableLambda(extract_order_id),
    original_feedback=RunnablePassthrough()
)

# 步骤2：并行分析任务
"""
返回的数据格式:
{
    "sentiment":{
        "sentiment":"NEGATIVE",                         # 情感类型
        "confidence":0.92,                              # 置信度
        "key_phrases":["物流太慢","承诺三天","实际七天"]    # 关键字
    },
    "categories":{
        "categories":["物流问题"]                        # 问题分类
    }
    "urgency":{
        "urgency":"HIGH",                               # 紧急程度
        "sla_hours":4,                                  # 响应时限(小时)
        "reason":"包含紧急处理要求"                        # 评估理由
    }
}
"""
# 节省大模型处理的时间,使用并行处理
analysis_chain = RunnableParallel(
    # 情感分析
    sentiment=RunnableLambda(analyze_sentiment),
    # 问题分类
    categories=RunnableLambda(classify_issue),
    # 紧急程度
    urgency=RunnableLambda(access_urgency)
)

# 步骤3：组合完整流程
process_chain = (
        extract_chain |
        RunnablePassthrough.assign(
            analysis=lambda x: analysis_chain.invoke(x["original_feedback"])
        ) |
        {
            "original_feedback": lambda x: x["original_feedback"],
            "order_id": lambda x: x["order_id"]["order_id"],
            "sentiment": lambda x: x["analysis"]["sentiment"].get("sentiment", "NEUTRAL"),
            "confidence": lambda x: x["analysis"]["sentiment"].get("confidence", 0.8),
            "key_phrases": lambda x: x["analysis"]["sentiment"].get("key_phrases", []),
            "categories": lambda x: x["analysis"]["categories"]["categories"],
            "urgency": lambda x: x["analysis"]["urgency"]["urgency"],
            "sla_hours": lambda x: x["analysis"]["urgency"]["sla_hours"],
            "urgency_reason": lambda x: x["analysis"]["urgency"].get("reason", ""),
        } |
        RunnableLambda(generate_response)
)

app = FastAPI(title="电商客服系统")


class FeedbackRequest(BaseModel):
    content: str
    user_id: str = "anonymous"


@app.get("/")
async def read_index():
    return FileResponse("index.html")


@app.post("/process-feedback")
async def process_feedback(feedback: FeedbackRequest):
    try:
        start = time.time()
        result = process_chain.invoke(feedback.content)
        process_time = time.time() - start

        return {
            "success": True,
            "process_time": process_time,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
