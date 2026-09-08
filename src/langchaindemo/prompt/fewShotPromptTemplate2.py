"""
少样本提示词模板
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from src.langchaindemo.model import getModel

model = getModel()

examples = [
    {"input": "如何重置密码?", "output": "密码重置可以通过绑定邮箱重置密码,也可以通过手机号重置密码"},
    {"input": "我的设备无法开机怎么办?",
     "output": "故障排除步骤:1.可能是遥控器电池没电,2.确认电源状态,3.确人设备是否被锁屏"},
    {"input": "这款产品是否有夜间模式?", "output": "这款不提供夜间模式,请选择xx款式的产品"}
]

examples_prompt_tmplt_txt = "用户问题：{input} 对应回答：{output}"
example_prompt = PromptTemplate.from_template(examples_prompt_tmplt_txt)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="你是一个智能客服，能够根据用户问题给出答案。",
    suffix="现在给你用户提问：{input}，请告诉我对应的结果。",
    input_variables=["input"],
)
fact_prompt = prompt.format(input="这款产品有防水模式？")
print(fact_prompt)
print('-' * 50)
result = model.invoke(fact_prompt)
print(result.content)
