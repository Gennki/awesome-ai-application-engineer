"""
注意事项：
API供应商可能有批量请求限制
输入列表中的所有字典必须有相同的键结构
批量处理不适合有状态的操作（如带记忆的对话链）
"""
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.langchaindemo.model import getModel

model = getModel()
prompt = PromptTemplate.from_template("用5句话来介绍{topic}")
parser = StrOutputParser()
chain = prompt | model | parser

topics = ["人工智能", "区块链", "量子计算", "基因编辑"]
input = [{"topic": topic} for topic in topics]

# 单词调用计时。串行执行，一个个主题执行
start = time.time()
single_result = [chain.invoke({"topic": topic}) for topic in topics]
single_time = time.time() - start

# 批量调用计时
start = time.time()
batch_result = chain.batch(input)
batch_time = time.time() - start

# 结果对比
print(f"\n=== 串行调用耗时：{single_time}s")
for i, res in enumerate(single_result):
    print(f"{topics[i]}: {res}")

print(f"\n=== 并行调用耗时：{batch_time}s")
for i, res in enumerate(batch_result):
    print(f"{topics[i]}: {res}")
