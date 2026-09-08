import datetime
import webbrowser

from langchain.agents import create_agent
from langchain_core.tools import tool

from src.langchaindemo.model import getModel


# 定义工具
# 工具下方的说明不能随便乱写，模型会根据说明来决定调用哪一个工具
@tool
def get_date():
    """获取今天的具体日期"""
    return datetime.date.today().strftime("%Y-%m-%d")


@tool
def open_browser(url: str, browser_name: str | None = None) -> str:
    """打开浏览器访问指定网址；调用前应确认网址和这个副作用。"""
    if browser_name:
        browser = webbrowser.get(browser_name)
    else:
        browser = webbrowser
    opened = browser.open(url)
    return f"浏览器打开请求已发送：{url}（成功={opened}）"


# 大模型客户端绑定工具
model = getModel()
agent = create_agent(
    model=model,
    tools=[get_date, open_browser],
)
# 执行agent
# result = agent.invoke({"messages": [{"role": "user", "content": "今天是几月几号？"}]})
# print(result)
# {'messages': [HumanMessage(content='今天是几月几号？', additional_kwargs={}, response_metadata={}, id='2b82cf01-f6f8-4093-bd92-eb4945a176e0'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 391, 'total_tokens': 434, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 15, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 7}, 'model_provider': 'openai', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': '424d17ed-bdf7-4cf1-8e63-fc21bc1f59d3', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--01a013c4-b4fb-7ff1-84ab-4940c8ef09da-0', tool_calls=[{'name': 'get_date', 'args': {}, 'id': 'call_00_gzQEpPklTH9mlv6WKnXQ0885', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 391, 'output_tokens': 43, 'total_tokens': 434, 'input_token_details': {'cache_read': 384}, 'output_token_details': {'reasoning': 15}}), ToolMessage(content='2026-08-18', name='get_date', id='859833d6-86b7-46d9-82a6-bf61bfbcb30d', tool_call_id='call_00_gzQEpPklTH9mlv6WKnXQ0885'), AIMessage(content='今天是 **2026年8月18日**，星期二。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 452, 'total_tokens': 466, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 68}, 'model_provider': 'openai', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'e39314c0-5392-410f-8069-ead3d7fb5ef0', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a013c4-b873-7d20-a34c-7b7bcbc3799a-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 452, 'output_tokens': 14, 'total_tokens': 466, 'input_token_details': {'cache_read': 384}, 'output_token_details': {'reasoning': 0}})]}
# 其中有个ToolMessage，表示调用了什么工具，这个工具返回了什么结果
# AIMessage中显示了最终大模型返回的结果：content='今天是 **2026年8月18日**，星期二。'

result = agent.invoke({"messages": [{"role": "user", "content": "帮我打开百度"}]})
print(result)
# {'messages': [HumanMessage(content='帮我打开百度', additional_kwargs={}, response_metadata={}, id='916a0dbd-786b-4816-822f-b0b09d791e9f'), AIMessage(content='我来帮你打开百度网站。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 77, 'prompt_tokens': 388, 'total_tokens': 465, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 22, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 4}, 'model_provider': 'openai', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': '51aaa028-c501-48bc-8801-ee150f83b33d', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--01a013c9-e00f-79e1-b398-8f7cc87ef15b-0', tool_calls=[{'name': 'open_browser', 'args': {'url': 'https://www.baidu.com'}, 'id': 'call_00_DrdGxLrhsNYFKhyYB8N80586', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 388, 'output_tokens': 77, 'total_tokens': 465, 'input_token_details': {'cache_read': 384}, 'output_token_details': {'reasoning': 22}}), ToolMessage(content='null', name='open_browser', id='89813112-e087-49d7-b1b5-c8deb904a09c', tool_call_id='call_00_DrdGxLrhsNYFKhyYB8N80586'), AIMessage(content='已经帮你打开了百度网站（www.baidu.com）。\n\n页面加载后，你就可以在搜索框中输入关键词进行搜索了。如果你还需要打开其他网站或执行其他操作，随时告诉我！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 55, 'prompt_tokens': 478, 'total_tokens': 533, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 16, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 94}, 'model_provider': 'openai', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'a6515ab5-7e7b-4fc3-9134-1bd0d0ec8d3b', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a013c9-e493-7023-ab9a-e696e63b3e30-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 478, 'output_tokens': 55, 'total_tokens': 533, 'input_token_details': {'cache_read': 384}, 'output_token_details': {'reasoning': 16}})]}
