"""
通过RunnableLambda实现默认Chain未实现的功能
"""
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, chain

from src.langchaindemo.model import getModel


# ============ 提供自定义函数 ============

# 输入一个字符串，返回它的字符长度
def length(text):
    return len(text)


# 输入两个字符串，返回它们长度的乘积
def multiple_length(text1, text2):
    return len(text1) * len(text2)


# 输入一个字典，从字典里取出 text1、text2，再交给 multiple_length 算乘积
# 作用：把"两个参数"包装成"一个字典"，方便在链里传值
#
# @chain 装饰器 = 把普通函数包装成 Runnable 的"第二种写法"
#   之前用 RunnableLambda(函数) 显式包装，现在用 @chain 直接写在函数头上
#   两者等价：装饰过之后，这个函数本身就是 Runnable，
#   可以直接用 | 连进 chain，不用再在外面套 RunnableLambda(...)
@chain
def multiple_length_by_dict(_dict):
    return multiple_length(_dict["text1"], _dict["text2"])


# ============ 1. 通过链来计算字符串长度 ============
# RunnableLambda(length) 把一个普通函数包成一个"流水线工位"
# invoke("Hello") 意思是：把 "Hello" 扔进工位，工位自动执行 length("Hello")
chain = RunnableLambda(length)
print(chain.invoke("Hello"))  # 输出：5

# ============ 2. 通过大模型计算两个字符串的长度的乘积 ============
# 三个基础组件：
#   prompt  : 模板。{a}、{b} 是占位符，等着被填入数值
#   model   : 大模型，负责真正"计算"
#   parser  : 把大模型的返回结果转成纯字符串
prompt = ChatPromptTemplate.from_template("{a} + {b} = ? 计算结果是多少？")
model = getModel()
parser = StrOutputParser()

chain = (
    # | 符号 = "流水线传送带"，把左边的东西递给右边处理
    #
    # 第一个 {} 是一个"拆盒工位"，输入的大盒子 {"k1":"Hello","k2":"World"} 会被
    # 同时送到左右两格（并行），每格独立处理，最后合成 {"a":..., "b":...}
    #
    # 下面左右两格正好演示了"把函数变成 Runnable"的两种等价写法：
    #   左格用【方式一：RunnableLambda(函数) 显式包装】
    #   右格用【方式二：@chain 装饰器 隐式包装】（见上方 multiple_length_by_dict）
    #
    # ── 左格：算出 a（方式一）──
    #   itemgetter("k1") 是一个"取物夹子"，伸进大盒子夹出标签 k1 的东西 → "Hello"
    #   "Hello" 经传送带递给 RunnableLambda(length)，算出长度 5
    #   结果标上标签 "a" → {"a": 5}
        {"a": itemgetter("k1") | RunnableLambda(length),

         # ── 右格：算出 b（方式二，这里还嵌套了一个"拆盒工位"）──
         #   里面又有一个 {}：先同时夹出 k1、k2 两样东西，
         #   重新装成一个小盒子 {"text1":"Hello", "text2":"World"}
         #   再经传送带递给 multiple_length_by_dict，从新盒子里取出 text1、text2，
         #   算长度的乘积 5 * 5 = 25
         #   注意：这里没有写 RunnableLambda(...)，
         #   因为函数头上已有 @chain，它本身就是 Runnable 了
         #   结果标上标签 "b" → {"b": 25}
         "b": ({"text1": itemgetter("k1"), "text2": itemgetter("k2")}
               | multiple_length_by_dict)} |
        # 拆盒工位输出 {"a": 5, "b": 25}
        # 经传送带递给 prompt，填进占位符 → "5 + 25 = ? 计算结果是多少？"
        prompt | model | parser
    # 再递给 model，由大模型回答
    # 最后经 parser 转成纯字符串
)
# 把输入盒子 {"k1":"Hello", "k2":"World"} 扔进整条流水线，从头走到尾，打印最终答案
print(chain.invoke({"k1": "Hello", "k2": "World"}))  # 输出：30
