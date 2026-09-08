"""
RunnablePassthrough：原样传递数据的"传送带"工位

一句话理解：它什么也不做，输入是什么，输出就是什么。
单独用它没用，它最大的价值在"并行"场景里，作为保留原始输入的那个分支。
"""
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# ============ 1. 它是什么：原样输出 ============
# 把它想象成一个"传送带"：东西放上去，不加工，原封不动送到对面
worker = RunnablePassthrough()
print(worker.invoke("你好"))             # 输出：你好
print(worker.invoke({"name": "张三"}))   # 输出：{'name': '张三'}


# ============ 2. 关键问题：原样输出就原样输出，为什么要专门做个组件？ ============
# 因为 LCEL 有条规矩：能放进 chain、能被 .invoke() 调用的，必须是一个 Runnable。
# 普通数据（字符串、字典）只是"原材料"，没有 .invoke() 方法，不能当 chain 里的节点。
#
# 所以 RunnablePassthrough 不是"原样输出的那个值"，而是
# "执行原样输出这个动作的那个节点"——就像传送带也是流水线的一环，只是不加工。
#
# 如果你不用它，就得自己写 lambda x: x 这样的函数，再靠 LangChain 自动包装，
# 代码又难看、意思又不清楚。它就是官方给"原样返回"这个动作的现成组件。


# ============ 3. 在并行里的价值：给原始输入留一条"旁路" ============
# RunnableParallel（并行）的特点：每个分支都收到同一份输入，
# 结果打包成字典：{"分支名": 该分支的结果}
#
# 有的分支要加工，有的分支想"原样保留输入"——
# 这时就把 RunnablePassthrough 放在那个不加工的分支里。
chain = RunnableParallel(
    passed=RunnablePassthrough(),   # 分支"passed"：原样透传，输入是什么就放什么
)
print(chain.invoke("Hello World"))
# 输出：{'passed': 'Hello World'}
# 解释：输入的 "Hello World" 被原封不动地放进了字典的 "passed" 键里


# ============ 4. .assign() 数据增强：保留原有的，再添加新的 ============
# 输入一个字典，输出还是这个字典，但额外多出几个新键
# 新键的值可以写死，也可以写一个 lambda 函数（接收整个输入字典）
chain = RunnablePassthrough().assign(
    modified=lambda x: x["k1"] + "!!!",   # lambda x: x 是输入字典，取出 x["k1"] 拼上 "!!!"
)
print(chain.invoke({"k1": "Hello World"}))
# 输出：{'k1': 'Hello World', 'modified': 'Hello World!!!'}
# 解释：原来的 k1 还在，只是多了一个由函数算出来的 modified


# ============ 5. 实际场景：RAG 问答的前半段 ============
# 场景：用户提问，我们要同时拿到「原问题」和「检索出的资料」，稍后再拼给大模型
# 这里 simulate_retrieve 只是模拟检索，真实项目里会换成真正的检索器
def simulate_retrieve(question):
    return f"关于「{question}」的参考资料..."


chain = RunnableParallel(
    question=RunnablePassthrough(),   # 原问题：原样透传
    context=simulate_retrieve,        # 资料：对同一个输入做加工
)
print(chain.invoke("什么是LCEL？"))
# 输出：{'question': '什么是LCEL？', 'context': '关于「什么是LCEL？」的参考资料...'}
