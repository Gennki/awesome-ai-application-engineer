from langchain_core.runnables import (
    RunnableLambda,
    RunnableMap,
    RunnableParallel,
    RunnableSequence,
)


def add_one(x: int) -> int:
    return x + 1


def mul_two(x: int) -> int:
    return x * 2


def mul_three(x: int) -> int:
    return x * 3


# ================= RunnableParallel 与 RunnableMap =================
# RunnableMap 是 RunnableParallel 的别名（源码里就是 RunnableMap = RunnableParallel），
# 两者完全等价、是同一个类。官方推荐统一用 RunnableParallel（与 JS 端命名一致）。
# 二者都用于「并行」组合：各分支同时执行、共享同一份输入，
# 输出以 {分支名: 结果} 的字典形式返回。

chain = RunnableParallel(
    a=add_one,
    b=mul_two,
    c=mul_three,
)
print(chain.invoke(1))  # {'a': 2, 'b': 2, 'c': 3}

chain = RunnableMap(
    a=add_one,
    b=mul_two,
    c=mul_three,
)
print(chain.invoke(1))  # {'a': 2, 'b': 2, 'c': 3}


# ================= RunnableSequence 与 RunnableParallel 的区别 =================
# RunnableSequence（串行）：节点按顺序执行，前一个的输出作为后一个的输入，
#                         整个链最终返回单一结果。
# RunnableParallel（并行）：各分支同时执行、共享同一份输入，
#                         最终返回一个 dict，键为各分支名。
#
# | 操作符等价于 RunnableSequence 串行组合，RunnableParallel / RunnableMap
# 用于并行组合，两者可以自由嵌套。

seq = RunnableSequence(RunnableLambda(add_one), RunnableLambda(mul_two))
print(seq.invoke(1))  # 1 -> add_one -> 2 -> mul_two -> 4

chain = RunnableLambda(add_one) | RunnableMap(
    a=mul_two,
    b=mul_three,
)
print(chain.invoke(2))  # 2 -> add_one -> 3，再并行 -> {'a': 6, 'b': 9}
