"""
少样本提示词模板
"""
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from src.langchaindemo.model import getModel

model = getModel()

examples = [
    {"input": "2+2", "output": "4", "description": "加法运算"},
    {"input": "5-2", "output": "3", "description": "减法运算"},
]
examples_prompt_tmplt_txt = "算式：{input} 值：{output} 类型：{description}"
example_prompt = PromptTemplate.from_template(examples_prompt_tmplt_txt)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="你是一个数学专家，能够准确说出算式类型。",
    suffix="现在给你算式：{input}，值{output}，告诉我类型",
    input_variables=["input", "output"],
)
fact_prompt = prompt.format(input="2*5", output="10")
print(fact_prompt)
print('-' * 50)
result = model.invoke(fact_prompt)
print(result.content)
