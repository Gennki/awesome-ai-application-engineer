import datetime
import webbrowser

from langchain_core.tools import tool

from src.langchaindemo.model import getModel

model = getModel()

# 不使用任何工具，直接调用大模型，返回的是错误的日期，或者直接返回无法查看当前日期
# content='今天是2026年6月25日，星期四。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 220, 'prompt_tokens': 89, 'total_tokens': 309, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 208, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 89}, 'model_provider': 'openai', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'a0ac9ae7-3741-414b-b100-6ff6472191c9', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a013b8-111e-7d40-a2f1-7a551df8e8eb-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 89, 'output_tokens': 220, 'total_tokens': 309, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 208}}
resp = model.invoke("今天是几月几号？")
print(resp)
print("=" * 100)


# 定义工具
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


# 绑定工具
tool_llm = model.bind_tools([get_date, open_browser])
# 返回内容如下：
# content='' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 44, 'prompt_tokens': 391, 'total_tokens': 435, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 16, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 7}, 'model_provider': 'openai', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': '0669ef34-1a31-4eaa-8a7e-11ab04caba30', 'finish_reason': 'tool_calls', 'logprobs': None} id='lc_run--01a013b8-1ac7-7171-98df-52d9ea4673c1-0' tool_calls=[{'name': 'get_date', 'args': {}, 'id': 'call_00_HmEesqXRAXNNj0hfhKDj0112', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 391, 'output_tokens': 44, 'total_tokens': 435, 'input_token_details': {'cache_read': 384}, 'output_token_details': {'reasoning': 16}}
# 注意其中的tool_calls=[{'name': 'get_date', 'args': {}, 'id': 'call_00_HmEesqXRAXNNj0hfhKDj0112', 'type': 'tool_call'}]
# 模型没有返回content，只生成了调用get_date工具的请求，并不会真正执行get_date的代码
resp = tool_llm.invoke("今天是几月几号？")
print(resp)
print("=" * 100)
