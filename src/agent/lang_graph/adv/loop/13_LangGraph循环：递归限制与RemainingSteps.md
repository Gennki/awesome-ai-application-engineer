# LangGraph 循环：用递归限制保护流程，用 RemainingSteps 提前收尾

前面的 [初识 Graph](../../base/09_初识Graph：State、Node与Edge.md) 介绍了固定边和条件边，[图可视化](../10_图可视化：Graphviz与Mermaid.md) 中的工具调用流程也会反复回到模型节点。这一篇继续回答两个问题：**怎样让图重复执行一组步骤？如果一直达不到业务结束条件，怎样停止并拿到已有结果？**

比如“生成草稿 → 检查质量 → 不合格就重写”，既需要正常完成的出口，也需要限制尝试次数。否则一次不理想的判断就可能让流程一直运行。

当前目录只有两个 Python 文件，但要分开看三个场景：

| 示例 | 代码当前状态 | 学习重点 |
| --- | --- | --- |
| [示例5.1_设置递归限制.py](示例5.1_设置递归限制.py) | 使用默认配置执行；显式设置 `4` 的调用被注释 | 列表累计到阈值后退出，以及超限时捕获异常 |
| [示例5.3_如何在达到递归限制前返回状态.py](示例5.3_如何在达到递归限制前返回状态.py) 前半段 | 整段被注释 | 结束条件始终不成立的循环 |
| 同一文件后半段 | 当前实际执行 | 用 `RemainingSteps` 在预算耗尽前正常返回状态 |

本文输出在项目虚拟环境中的 **LangGraph 1.2.10** 核对，递归限制的边界结果以该环境为准。两个示例不调用模型，不需要 API 密钥；原文件执行前会生成 Mermaid 图片，离线验证时需要跳过绘图，方法见第 7 节。

## 知识点目录

1. [循环的基本结构：条件边与回边](#1-循环的基本结构条件边与回边)
2. [示例5.1：状态怎样推动循环结束](#2-示例51状态怎样推动循环结束)
3. [recursion_limit：限制的是图步骤](#3-recursion_limit限制的是图步骤)
4. [示例5.3：用 RemainingSteps 提前返回](#4-示例53用-remainingsteps-提前返回)
5. [剩余步骤与收尾预算：为什么使用阈值 2](#5-剩余步骤与收尾预算为什么使用阈值-2)
6. [常见疑问与容易混淆的细节](#6-常见疑问与容易混淆的细节)
7. [运行与练习：离线验证正常路径和边界](#7-运行与练习离线验证正常路径和边界)
8. [按需求选择循环的退出方式](#8-按需求选择循环的退出方式)

---

## 1. 循环的基本结构：条件边与回边

本目录的循环由图上的边组成，不需要在节点函数中写 `while True`。示例5.1的连接关系是：

```text
START → a → route
        ↑     ├─ aggregate 长度 < 7 → b ─┐
        └───────────────────────────────┘
              └─ aggregate 长度 ≥ 7 → END
```

`route` 是附着在 `a` 后面的条件函数，并没有通过 `add_node()` 注册成独立节点。为了说明判断过程，图中把它单独列出。

对应的三条连接是：

```python
builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
```

执行顺序是：

1. `a` 读取状态，返回本步更新。
2. 框架按 reducer 合并更新，`route` 读取更新后的状态。
3. `route` 返回 `"b"` 就继续，返回 `END` 就结束。
4. `b` 完成后固定回到 `a`，再次更新并判断。

因此，循环是否有出口不能只看有没有 `END`，还要检查通向 `END` 的条件能否真的成立。

这里也要区分三种“循环”：

| 写法 | 重复执行什么 | 本篇的递归预算怎样作用 |
| --- | --- | --- |
| 图上的 `b → a` 回边 | 图运行时重复调度节点 | 会继续消耗图步骤预算 |
| 节点函数内部的 Python `while` | 同一次节点调用里的代码 | 不会因为每次 `while` 迭代自动扣减图步骤 |
| 外层 `for` 消费 `graph.stream()` | 调用方逐条读取事件 | 消费事件本身不等于新增一次图步骤 |

如果一个节点内部的 `while` 一直不返回，不能指望 `recursion_limit` 像超时器一样打断它。它也不是运行秒数、Token 数量或模型调用费用上限。

> **小结：** 回边提供重复执行的路径，条件边提供出口。预算控制的是图的调度过程。

---

## 2. 示例5.1：状态怎样推动循环结束

### 2.1 完整示例：每步追加一个标记

下面保留原文件的节点和路由逻辑，移除绘图，并显式设置足够的预算，方便独立运行：

```python
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    aggregate: Annotated[list[str], operator.add]


def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}


def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}


def route(state: State) -> Literal["b", "__end__"]:
    if len(state["aggregate"]) < 7:
        return "b"
    return END


builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()

try:
    result = graph.invoke({"aggregate": []}, config={"recursion_limit": 8})
    print(result)
except GraphRecursionError:
    print("Recursion Error")
```

原文件的返回注解是 `Literal["b", END]`。这里用 `END` 对应的字符串字面量 `"__end__"` 描述可能的返回值；函数运行时仍然返回 `END`。注解帮助描述可选目标，实际去哪里由函数返回值决定。

### 2.2 Reducer 是计数能够增长的前提

状态声明中的关键部分是：

```python
aggregate: Annotated[list[str], operator.add]
```

列表上的 `operator.add` 表示拼接：

```text
旧状态：    ["A", "B"]
a 返回：    ["A"]
合并后：    ["A", "B", "A"]
```

节点应返回本步增量 `{"aggregate": ["A"]}`，不应把旧列表也放进更新。若返回 `state["aggregate"] + ["A"]`，reducer 会再次拼接旧值，导致重复累计。

反过来，如果把字段改成普通 `aggregate: list[str]`，新的单元素列表会覆盖旧列表。本例的长度就总是 `1`，无法达到退出阈值，最后只能由硬限制截断。

### 2.3 从空列表到最终状态

| 执行次序 | 节点 | 更新后的 `aggregate` | 后续路径 |
| ---: | --- | --- | --- |
| 1 | `a` | `["A"]` | 长度 1，去 `b` |
| 2 | `b` | `["A", "B"]` | 固定回 `a` |
| 3 | `a` | `["A", "B", "A"]` | 长度 3，去 `b` |
| 4 | `b` | `["A", "B", "A", "B"]` | 固定回 `a` |
| 5 | `a` | `["A", "B", "A", "B", "A"]` | 长度 5，去 `b` |
| 6 | `b` | `["A", "B", "A", "B", "A", "B"]` | 固定回 `a` |
| 7 | `a` | `["A", "B", "A", "B", "A", "B", "A"]` | 长度 7，去 `END` |

打印发生在节点返回更新之前，所以最后一条节点日志只有 6 个元素，最终结果才有 7 个：

```text
Node A sees ['A', 'B', 'A', 'B', 'A', 'B']
{'aggregate': ['A', 'B', 'A', 'B', 'A', 'B', 'A']}
```

结束判断只在 `a` 后执行。若把阈值从 `7` 改成 `6`，流程仍会在第 7 个元素写入后退出；`b` 写入第 6 个元素时没有条件判断。这是判断位置导致的结果，不是 reducer 失效。

> **小结：** 先看字段如何合并，再看路由在哪个节点之后检查。累计长度在本例中记录的是节点执行次数，不是完整循环轮数。

---

## 3. recursion_limit：限制的是图步骤

### 3.1 配置放在顶层

原文件中这行被注释了：

```python
result = graph.invoke({"aggregate": []}, {"recursion_limit": 4})
```

启用它之前应替换原来的调用，否则会先跑一次默认配置，再跑一次小预算配置。也可以明确写出参数名：

```python
result = graph.invoke(
    {"aggregate": []},
    config={"recursion_limit": 4},
)
```

`recursion_limit` 是运行配置的顶层字段，不能放进状态，也不要放到 `configurable` 内。如果另一个图确实配置了 Checkpointer，可以把两类配置并列：

```python
config = {
    "recursion_limit": 20,
    "configurable": {"thread_id": "loop-001"},
}
```

本目录两个图都没有 Checkpointer，不需要 `thread_id`。仅添加一个会话编号，也不会自动保存状态。

### 3.2 Super-step 与循环轮数不同

图的预算按 super-step（超级步）推进。同一个超级步可以有多个并行节点；串行节点则依次推进。本目录是简单串行循环，因此可以先按“一个节点执行对应一次推进”理解。

例如 `decision → action → decision` 包含三次串行节点执行，不能当成一次业务循环只消耗一个步骤。反过来，若某一步同时运行两个并行节点，也不应直接把预算消耗算成两步。

`route` 在所属节点的执行过程中决定后续目标，本例并不额外占据一个独立节点步骤。`START`、`END` 也不是用户实现的业务节点；不要仅靠给静态图中的每个框计数来确定最小预算。

### 3.3 超限会抛异常，不会自动返回最终状态

把第 2 节的调用改为限制 `4`，可以看到：

```text
Node A sees []
Node B sees ['A']
Node A sees ['A', 'B']
Node B sees ['A', 'B', 'A']
Recursion Error
```

这时业务退出条件还没满足。`GraphRecursionError` 来自图运行时，与 Python 递归调用产生的 `RecursionError` 不同；调整 `sys.setrecursionlimit()` 不能解决这里的问题。

对于当前环境，从空列表开始运行的边界结果是：

| `recursion_limit` | 观察到的节点执行 | 调用结果 |
| ---: | --- | --- |
| `4` | `a → b → a → b` | 抛 `GraphRecursionError` |
| `7` | 7 个节点均已执行，末次路由选择 `END` | 仍抛 `GraphRecursionError` |
| `8` | 7 个节点执行完成 | 正常返回 7 个元素 |

**第 7 个节点选择了 `END`，不代表限制恰好设为 7 就一定成功。** 本地版本在预算边界还有运行完成的检查，因此本例需要留出余量。不要把表中的差一现象推广为所有版本、所有图都必须“节点数加一”；应在实际环境验证，并避免把预算压到临界值。

原文件没有显式设置预算时使用运行时默认值。默认值可能随版本和依赖变化，本文的边界实验始终显式设置，不依赖“默认一定是 25”之类的记忆。

### 3.4 `except` 不会把异常转换成部分结果

当 `graph.invoke()` 抛异常时：

```python
result = graph.invoke(...)
```

这次赋值不会完成。如果之前没有定义 `result`，就没有本次调用的结果；如果它以前有值，也不能把旧值当成本次结果。原文件的 `except` 只是打印 `Recursion Error`。

需要部分进度时，可以在调用方消费 `stream_mode="values"` 的状态快照，并在异常发生时保留最后一次收到的快照。若要在图内正常返回，则使用下一节的主动退出方式。通过 Checkpointer 查询状态是另一种方案，但需要先配置持久化，不能假设本目录已具备该能力。

> **小结：** 业务条件负责正常完成，递归限制是兜底。抛异常与主动返回 `END` 是不同的结果路径。

---

## 4. 示例5.3：用 RemainingSteps 提前返回

### 4.1 不带 RemainingSteps 的版本为什么停不下来

原文件前半段被注释，其核心逻辑是：

```python
def decision_node(state):
    return {"value": "keep going!"}


def router(state):
    if state["value"] == "end":
        return END
    return "action"
```

图的连接是 `START → decision → action → decision`，`action` 只写入 `action_result`。每次判断前，`decision` 都把 `value` 改成 `"keep going!"`，所以业务结束条件永远不成立。

连输入 `{"value": "end"}` 也没有用：它会先经过 `decision`，再进入 `router`，判断前就被覆盖了。把 `recursion_limit` 调大，只会让这段循环执行更多次。

### 4.2 增加一个由运行时管理的字段

活动版本多了这一项：

```python
from langgraph.managed.is_last_step import RemainingSteps


class State(TypedDict):
    value: str
    action_result: str
    remaining_steps: RemainingSteps
```

`remaining_steps` 是托管值（managed value），由 LangGraph 根据运行进度提供。读取时得到整数，但声明时应使用 `RemainingSteps`，不是普通 `int`。

在本地实现中，它根据运行时的停止边界与当前步骤之差计算。业务节点不需要自己减一，调用方也不需要传入这个字段。它表示剩余的图步骤预算，不是剩余秒数或剩余 `action` 次数。

### 4.3 完整示例：小预算下观察主动退出

下面保留当前代码的处理逻辑，把预算固定为 `6`，并增加路由日志：

```python
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.managed import RemainingSteps


class State(TypedDict):
    value: str
    action_result: str
    remaining_steps: RemainingSteps


def decision_node(state: State):
    return {"value": "keep going!"}


def action_node(state: State):
    return {"action_result": "what a great result!"}


def router(state: State) -> Literal["action", "__end__"]:
    print(f'remaining_steps={state["remaining_steps"]}')
    if state["remaining_steps"] <= 2:
        return END
    if state["value"] == "end":
        return END
    return "action"


workflow = StateGraph(State)
workflow.add_node("decision", decision_node)
workflow.add_node("action", action_node)
workflow.add_edge(START, "decision")
workflow.add_conditional_edges("decision", router, ["action", END])
workflow.add_edge("action", "decision")
app = workflow.compile()

result = app.invoke({"value": "hi!"}, config={"recursion_limit": 6})
print(result)
```

这里从 `langgraph.managed` 导入 `RemainingSteps`；原文件从 `langgraph.managed.is_last_step` 导入。在本地环境中两者指向同一托管类型。

输出为：

```text
remaining_steps=5
remaining_steps=3
remaining_steps=1
{'value': 'keep going!', 'action_result': 'what a great result!'}
```

实际路径是：

```text
START
  → decision（route 看到剩余 5）
  → action
  → decision（route 看到剩余 3）
  → action
  → decision（route 看到剩余 1，返回 END）
  → END
```

两次路由判断之间经过 `action` 和 `decision`，所以日志中的剩余预算每次减少 `2`，而不是减少 `1`。

`action_result` 是普通字符串字段，没有追加 reducer。两次 `action` 都写入同一个字符串，最终状态只保存一次这个值。只看最终结果，无法判断 `action` 执行了几次；需要结合事件或日志观察。

### 4.4 最终输出里为什么没有 remaining_steps

托管字段供运行中的节点与路由读取，不是普通业务输出通道。当前图的最终结果只包含已写入的 `value`、`action_result`，不会自动带上 `remaining_steps`。

如果调用方需要知道“因为业务完成退出，还是因为预算不足退出”，应增加普通字段，例如 `stop_reason`，由节点返回更新。条件函数这里只返回下一目标，并没有写入退出原因。

> **小结：** `RemainingSteps` 让图在运行过程中感知预算，再主动走向 `END`。正常返回已有状态不等于业务任务已经完成。

---

## 5. 剩余步骤与收尾预算：为什么使用阈值 2

### 5.1 当前检查点每隔两个节点才出现一次

当前图中，路由如果继续，就会走：

```text
本次 decision 的 route → action → 下一次 decision 的 route
```

只有再次到达 `decision`，才会检查预算。`<= 2` 的含义是：预算已经很紧时，不再启动下一轮，而是在当前检查点直接结束。

如果只在 `remaining_steps == 0` 时退出，就可能来不及再次走到这个判断位置。使用 `<=` 也能覆盖不同预算起点下观察到的奇数或偶数。

不要把 `2` 记成所有图通用的常量。若继续路径增加了更多串行节点，或者退出前还要运行汇总节点，就需要更早判断，并给收尾路径预留空间。

### 5.2 小预算揭示“正常返回”与“业务成功”的区别

下面是活动版本从 `{"value": "hi!"}` 开始运行的结果：

| `recursion_limit` | 执行路径 | 结果 |
| ---: | --- | --- |
| `1` | `decision` 已执行并选择结束 | 仍触发 `GraphRecursionError`，预算过紧 |
| `2` | `decision → END` | `{"value": "keep going!"}` |
| `3` | `decision → END` | `{"value": "keep going!"}` |
| `4` | `decision → action → decision → END` | 包含 `value` 和 `action_result` |
| `6` | 两次 `action`，三次 `decision` 后结束 | 包含 `value` 和 `action_result` |

有两个重要结论：

- `RemainingSteps` 不会让任意小的预算都成功；本地环境的限制 `1` 仍会报错。
- 限制为 `2`、`3` 时，`action` 根本没有执行，最终结果没有 `action_result`。`TypedDict` 中写了字段，不会自动创建它或填入默认值。

如果允许返回部分结果，调用方可以用 `result.get("action_result")` 检查；如果业务要求必须产生结果，应为必要节点保留足够预算，并明确处理预算不足的情况。

### 5.3 给降级结果一个明确原因

下面是替换第 4.3 节状态、节点和路由定义的示意，建图连接可以保持不变：

```python
class State(TypedDict):
    value: str
    action_result: str
    stop_reason: str
    remaining_steps: RemainingSteps


def decision_node(state: State):
    if state["value"] == "end":
        return {"stop_reason": "completed"}
    if state["remaining_steps"] <= 2:
        return {"stop_reason": "step_budget"}
    return {"value": "keep going!", "stop_reason": "running"}


def router(state: State) -> Literal["action", "__end__"]:
    if state["stop_reason"] in {"completed", "step_budget"}:
        return END
    return "action"
```

这是业务扩展示意，原文件没有这个字段。它把业务完成放在预算判断之前，让调用方能够区分两种退出原因。输入 `value="end"` 时会保留原值并报告完成；其他输入仍由预算分支兜底。

如果还要生成总结，可以再增加一个汇总节点，但应相应提前触发收尾，而不是在预算已经耗尽时才安排它。

> **小结：** 判断阈值取决于离下一个安全出口还有多远。给用户返回已有状态时，应明确它是完整结果还是预算不足时的部分结果。

---

## 6. 常见疑问与容易混淆的细节

### 6.1 调大限制就能修复无限循环吗？

不能。如果状态永远不会满足终止条件，调大限制只会增加重复次数。先检查路由、节点是否覆盖判断字段，以及 reducer 是否让计数按预期变化；确认是合法的长流程后，再增加预算。

### 6.2 `RemainingSteps` 能不能手动传入或更新？

不应把它当普通业务计数器。不要在输入中设置 `remaining_steps=100`，也不要让节点返回 `{"remaining_steps": ...}` 来修改预算。改变预算应使用顶层 `recursion_limit`；需要“最多尝试三次”这样的业务约束，则另设普通字段 `attempts` 或 `max_attempts`。

图步骤与业务尝试次数可以同时存在：一次尝试可能包含检索、生成、检查三个串行节点。

### 6.3 再调用一次 invoke 会接着上次状态继续吗？

本目录的两张图都是直接 `compile()`，没有配置 Checkpointer。每次提供新输入调用，都会开始一次独立运行。一次因递归限制报错后，再次调用相同输入，并不意味着从失败位置恢复。

暂停和恢复可以结合 [中断与编辑状态](../interr/11_LangGraph中断与编辑状态.md) 学习。`END` 是结束本次运行，`interrupt()` 是带状态保存的交互暂停，两者用途不同。

### 6.4 `add_conditional_edges` 后为什么没有再写到 END 的边？

`route` 或 `router` 返回 `END` 时，条件路由已经表达了退出路径，不需要再给同一源节点添加一条无条件结束边。

同样，不要在保留条件边的同时又给 `decision` 添加固定的 `decision → action` 边。固定边与条件边可以共同安排任务；如果想表达“继续或结束二选一”，应让一个清晰的路由负责选择。

### 6.5 为什么源文件的 try 没有捕获绘图错误？

两个文件的 `draw_mermaid_png()` 都在 `try` 之前执行，而且捕获类型只有 `GraphRecursionError`。绘图网络失败、输出目录不存在，不属于递归错误，可能在图真正运行前就终止脚本。

当前活动代码的图像路径是 `../../../../../assets/示例5.1.png` 和 `../../../../../assets/示例5.3.png`。这些路径相对于**启动 Python 时的工作目录**解释，不是自动相对于 `.py` 文件所在目录；只有工作目录为本篇所在的 `loop/` 时，才会指向仓库根目录的 `assets/`。验证循环时可以先跳过绘图，不必先把图片生成环境配置好。

---

## 7. 运行与练习：离线验证正常路径和边界

### 7.1 先确认本地版本

从项目根目录执行：

```bash
.venv/bin/python -c 'from importlib.metadata import version; print(version("langgraph"))'
```

第 2.1、4.3 节是可独立运行的完整代码，都已移除绘图；其余短片段是对应上下文中的说明或替换片段。

若直接运行原文件：

```bash
.venv/bin/python src/agent/lang_graph/adv/loop/示例5.1_设置递归限制.py
.venv/bin/python src/agent/lang_graph/adv/loop/示例5.3_如何在达到递归限制前返回状态.py
```

需要先处理第 6.5 节的绘图依赖与路径。为了只验证循环，推荐下面不修改源码的方式。

### 7.2 跳过绘图，断言四类结果

在仓库根目录运行以下命令。`runpy.run_path()` 会执行文件顶层代码；加载时暂时拦截绘图，并收起原脚本演示的打印，再对编译后的图进行独立检查：

```bash
.venv/bin/python - <<'PY'
import contextlib
import io
import runpy
from pathlib import Path
from unittest.mock import patch

from langchain_core.runnables.graph import Graph
from langgraph.errors import GraphRecursionError

loop_dir = Path("src/agent/lang_graph/adv/loop")


def load_example(filename):
    with patch.object(Graph, "draw_mermaid_png", return_value=b""):
        with contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(str(loop_dir / filename))


graph = load_example("示例5.1_设置递归限制.py")["graph"]
app = load_example("示例5.3_如何在达到递归限制前返回状态.py")["app"]

# 1. 正常业务出口：足够预算时累计到 7。
with contextlib.redirect_stdout(io.StringIO()):
    result = graph.invoke({"aggregate": []}, config={"recursion_limit": 8})
assert result == {"aggregate": ["A", "B", "A", "B", "A", "B", "A"]}
print("正常结束：", result)

# 2. 硬限制：预算不足，以及当前版本的临界值。
for limit in (4, 7):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            graph.invoke({"aggregate": []}, config={"recursion_limit": limit})
    except GraphRecursionError:
        print(f"限制 {limit}：捕获 GraphRecursionError")
    else:
        raise AssertionError(f"限制 {limit} 本应触发递归错误，请检查版本")

# 3. 主动退出：保留已产生的 action_result，托管字段不在输出里。
result = app.invoke({"value": "hi!"}, config={"recursion_limit": 6})
assert result == {
    "value": "keep going!",
    "action_result": "what a great result!",
}
assert "remaining_steps" not in result
print("预算内返回：", result)

# 4. 很小的预算：action 没有执行，所以返回部分状态。
partial = app.invoke({"value": "hi!"}, config={"recursion_limit": 2})
assert partial == {"value": "keep going!"}
print("部分状态：", partial)

# 原代码会覆盖输入的 end，并不是通过业务完成分支退出。
assert app.invoke({"value": "end"}, config={"recursion_limit": 6}) == result
print("全部断言通过")
PY
```

这段验证不需要网络，也不创建图片。两个源文件不设置 Checkpointer，加载时的默认调用不会把业务状态累计到后续断言中。

### 7.3 观察执行次数，而不只看最终结果

在第 4.3 节建好的 `app` 后运行：

```python
for event in app.stream(
    {"value": "hi!"},
    config={"recursion_limit": 6},
    stream_mode="updates",
):
    print(event)
```

除路由中的日志外，节点更新事件按顺序是：

```text
{'decision': {'value': 'keep going!'}}
{'action': {'action_result': 'what a great result!'}}
{'decision': {'value': 'keep going!'}}
{'action': {'action_result': 'what a great result!'}}
{'decision': {'value': 'keep going!'}}
```

这里的 `stream()` 会重新运行图，不是查看上一轮 `invoke()` 的录像。`updates` 展示的是各节点提交的增量；要看合并后的业务状态，可以使用 `stream_mode="values"`。

### 7.4 三个小练习

1. 将示例5.1的长度阈值从 `7` 改为 `6`，先预测最终长度，再检查为什么仍是 `7`。
2. 在第 4.3 节中分别使用预算 `2`、`4`、`6`，统计 `action` 执行次数，对照最终字段是否存在。
3. 使用第 5.3 节的替换定义，分别输入 `"end"` 和 `"hi!"`，检查 `stop_reason` 能否区分业务完成与预算退出。

每次只改变一个条件，更容易分清变化来自状态合并、路由位置还是运行预算。

---

## 8. 按需求选择循环的退出方式

| 需求 | 使用方式 | 需要注意 |
| --- | --- | --- |
| 达到业务条件后结束 | 条件边返回 `END` | 条件必须可达，并能看到更新后的状态 |
| 避免异常长时间重复调度 | 顶层 `recursion_limit` | 是图步骤上限，超限会抛异常 |
| 接近预算时返回已有结果 | `RemainingSteps` 加主动路由 | 为后续必要步骤留出余量 |
| 区分成功与部分完成 | 普通状态字段 `stop_reason` | 由节点写入，正常返回不等于业务成功 |
| 限制最多尝试几次 | 普通业务计数器 | 尝试次数与图步骤分别管理 |
| 观察到底重复了几次 | `stream()` 事件或节点日志 | 最终覆盖型字段不能保留执行历史 |

排查时可以沿着这条线索逐项检查：

```text
节点更新是否正确
  → reducer 是否保留了进展
  → 条件路由是否检查更新后的状态
  → 通向 END 的条件是否可达
  → 正常路径和收尾路径是否有足够预算
  → 调用方能否识别部分结果
```

业务条件决定“任务是否完成”，`RemainingSteps` 决定“预算是否允许继续”，`recursion_limit` 提供最后的执行上限。学习本目录时，应把这三个职责一起理解，而不是只记住一个参数名称。
