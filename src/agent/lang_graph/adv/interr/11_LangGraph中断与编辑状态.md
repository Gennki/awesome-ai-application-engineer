# LangGraph 中断与编辑状态：让工作流暂停、等待人类决定，再从断点继续

我们写 LangGraph 时，最容易上手的是“一条流水线”：节点按顺序执行，最后返回结果。可一旦流程走进真实业务，事情马上变得不一样了：删除文件前要问一句“确定吗？”，表单输入错了要重新填写，模型生成草稿后还要允许运营同学改两个字。

这篇笔记围绕这几个场景展开。代码来自当前目录下的四个示例，我们先暂停、再恢复，接着处理多次交互和状态编辑，最后把人工问答接进 Agent 的工具调用流程。重点还会拆开一个容易混淆的问题：图已经中断了，为什么 Python 还能继续执行下一个 `for`？

## 知识点目录

1. [先搞清楚：中断为什么需要 Checkpointer](#1-先搞清楚中断为什么需要-checkpointer)
2. [用 interrupt 实现人工审批](#2-用-interrupt-实现人工审批)
3. [在循环里反复中断：直到拿到合法输入](#3-在循环里反复中断直到拿到合法输入)
4. [用 interrupt_before 暂停节点，并编辑图状态](#4-用-interrupt_before-暂停节点并编辑图状态)
5. [在 Agent 中使用 interrupt：看懂两个 for 循环的执行顺序](#5-在-agent-中使用-interrupt看懂两个-for-循环的执行顺序)
6. [按需求选择：动态中断、静态断点和状态查询](#6-按需求选择动态中断静态断点和状态查询)
7. [最佳实践：把暂停流程做得可靠](#7-最佳实践把暂停流程做得可靠)

---

## 1. 先搞清楚：中断为什么需要 Checkpointer

先想象一个快递流程：包裹扫描到“等待人工确认”这一步，系统不能只打印一句提示就结束进程。它得把包裹当前在哪、已经做过什么、下一步要去哪儿保存下来，等人工点击按钮后再接着走。

LangGraph 里的 `interrupt()` 就是这个“等待人工确认”的按钮。它暂停的是一次图运行，而不是简单地从 Python 函数里 `return`；要想之后恢复，图必须在编译时配置 `checkpointer`，调用时还要提供 `thread_id`。`thread_id` 可以理解成这条状态历史的会话编号，同一个编号才能找到刚才那次暂停。

最简单的配置长这样：

~~~python
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    message: str


def greet(state: State):
    return {"message": state["message"] + "，欢迎来到 LangGraph"}


builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "order-1001"}}

print(graph.invoke({"message": "你好"}, config=config))
~~~

这里的 `MemorySaver` 适合学习和本地调试，它把检查点放在进程内存里。程序一重启，状态也就跟着下班了；生产环境要换成持久化的 checkpointer，并根据业务处理数据保留、权限和清理策略。

**你可能会想：只传 `thread_id` 不就够了吗？** 还不够。`thread_id` 只是“去哪个抽屉找记录”的钥匙，真正负责保存记录的是 `checkpointer`。两者缺一个，恢复流程就接不上。

> **小结：** `interrupt()` 负责暂停，`checkpointer` 负责保存，`thread_id` 负责定位同一条状态历史。

---

## 2. 用 interrupt 实现人工审批

先做一个最常见的例子：删除文件前让人确认。下面代码可以直接运行，它会先停在审批节点，打印中断内容；再用 `Command(resume=True)` 模拟用户点击“批准”。

### 完整示例：审批通过或取消

~~~python
from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]


def approval_node(
    state: ApprovalState,
) -> Command[Literal["proceed", "cancel"]]:
    decision = interrupt(
        {
            "question": "批准此操作？",
            "details": state["action_details"],
        }
    )
    # 恢复执行时，decision 就是 Command(resume=...) 传进来的值。
    return Command(goto="proceed" if decision else "cancel")


def proceed_node(state: ApprovalState):
    return {"status": "approved"}


def cancel_node(state: ApprovalState):
    return {"status": "rejected"}


builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "approval-1001"}}

paused = graph.invoke(
    {"action_details": "删除 /tmp/report.csv", "status": "pending"},
    config=config,
)
print("等待人工输入：", paused["__interrupt__"])

# 真实项目里，这里通常来自前端按钮或人工审核接口。
finished = graph.invoke(Command(resume=True), config=config)
print("最终状态：", finished["status"])
~~~

运行后，第一次 `invoke()` 不会走到 `proceed` 或 `cancel`，而是返回 `__interrupt__`。传给 `interrupt()` 的字典就是交给 UI 的展示数据，前端可以把 `question` 显示成标题，把 `details` 显示成待审核内容。

恢复时必须使用同一个 `config`，也就是同一个 `thread_id`。`Command(resume=True)` 的值会成为 `interrupt()` 的返回值，于是节点返回 `Command(goto="proceed")`，流程继续往下走。把 `True` 改成 `False`，结果就会变成 `rejected`；仓库里的原始示例最后一行注释写成了 `approved`，实际运行结果应以取消分支为准。

这里还有一个很容易踩的坑：恢复后，被中断的节点会从函数开头重新执行，直到再次走到原来的 `interrupt()`。因此，`interrupt()` 前面不要放“扣库存、发邮件、写数据库”这类不可重复的副作用；如果确实要做，就把它做成幂等操作，或者挪到中断之后。

> **小结：** 把需要人决定的地方放进 `interrupt()`，把展示信息放进它的参数，再用同一个 `thread_id` 和 `Command(resume=...)` 恢复。

---

## 3. 在循环里反复中断，直到拿到合法输入

审批通常只问一次，表单校验却可能要来回几轮。年龄输入就是一个很好的练习：用户第一次输入 `"thirty"`，程序提示格式错误；第二次输入 `"三十"`，仍然不接受；直到收到整数 `30`，节点才真正返回。

### 完整示例：交互式校验年龄

~~~python
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    age: int | None


def collect_age(state: FormState):
    prompt = "你几岁了？"

    while True:
        answer = interrupt(prompt)

        if isinstance(answer, int) and answer > 0:
            return {"age": answer}

        prompt = f"「{answer}」不是有效的年龄，请输入一个正整数。"


def show_result(state: FormState):
    print(f"收到年龄：{state['age']}")
    return {}


builder = StateGraph(FormState)
builder.add_node("collect_age", collect_age)
builder.add_node("show_result", show_result)
builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", "show_result")
builder.add_edge("show_result", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "form-1001"}}

result = graph.invoke({"age": None}, config=config)
print("第一次提问：", result["__interrupt__"])

result = graph.invoke(Command(resume="thirty"), config=config)
print("第二次提问：", result["__interrupt__"])

result = graph.invoke(Command(resume="三十"), config=config)
print("第三次提问：", result["__interrupt__"])

result = graph.invoke(Command(resume=30), config=config)
print("最终结果：", result["age"])
~~~

第一次执行到 `interrupt()` 时，节点暂停在循环里。每次恢复，节点都会从函数开头重新运行，但 LangGraph 会按顺序把之前提交过的恢复值交给对应的 `interrupt()` 调用，所以 `"thirty"`、`"三十"` 和 `30` 会依次被消费。你可以把它想成一叠待处理的回执：节点重新走到某个暂停点时，框架把这一轮对应的回执递给它。

**你可能会想：为什么不直接在节点里调用 `input()`？** 因为 `input()` 会把图和终端绑死，服务化后很难接网页、手机端或消息队列；`interrupt()` 把“暂停”变成图的能力，输入从哪里来由调用方决定。

实际应用中，`answer` 可能来自表单、客服人工坐席，也可能来自另一个系统。校验逻辑仍然留在图节点里，交互界面只负责展示 `interrupt` 的内容并把用户输入传回 `Command(resume=...)`。

> **小结：** 循环里的多个 `interrupt()` 可以做成表单校验、补充信息和重试流程；要记住节点会重跑，代码必须能安全重复执行。

---

## 4. 用 interrupt_before 暂停节点，并编辑图状态

前面的暂停点写在节点函数内部，适合“执行到某个业务问题后向人提问”。还有一种情况：我们希望节点还没开始执行，就先检查一下输入，必要时直接改状态。这时可以在编译图时配置 `interrupt_before=["step_2"]`。

下面示例让图先执行 `step_1`，在 `step_2` 开始前停住。我们读取当前快照，把 `input` 从英文改成中文，然后继续执行。

### 完整示例：暂停、查看、编辑、继续

~~~python
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    input: str


def step_1(state: State):
    print("step_1 收到：", state["input"])
    return {}


def step_2(state: State):
    print("step_2 收到：", state["input"])
    return {}


def step_3(state: State):
    print("step_3 收到：", state["input"])
    return {}


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["step_2"],
)
config = {"configurable": {"thread_id": "edit-1001"}}

for event in graph.stream(
    {"input": "hello world"},
    config,
    stream_mode="values",
):
    print("事件：", event)

snapshot = graph.get_state(config)
print("暂停时的状态：", snapshot.values)
print("暂停时的下一节点：", snapshot.next)

graph.update_state(config, {"input": "你好，宇宙！"})
print("编辑后的状态：", graph.get_state(config).values)

for event in graph.stream(None, config, stream_mode="values"):
    print("继续执行：", event)
~~~

`get_state(config)` 返回的是当前检查点快照，`values` 是状态值，`next` 能告诉我们下一步还剩哪些节点。`update_state()` 会在这条会话历史上写入一次状态更新；这里修改 `input` 后，下一节点仍然是 `step_2`。

最后用 `graph.stream(None, config, ...)` 继续，传入 `None` 的意思是“不要开启一轮新的输入，请从当前检查点接着跑”。执行结果会显示 `step_2` 和 `step_3` 都拿到了修改后的“你好，宇宙！”。仓库中的原始代码还调用了 `draw_mermaid_png()`，这一步需要访问 Mermaid 服务，离线运行时可以先删掉或改用本地渲染，不影响状态编辑本身。

**你可能会想：`interrupt_before` 和 `interrupt()` 到底差在哪儿？** 前者是图级别的静态断点，节点还没有执行；后者是节点内部的动态暂停，节点可以根据业务数据决定何时停、停几次。一个像调试器的断点，一个像流程中的“请用户回答”。

> **小结：** `interrupt_before` 适合人工检查和改写状态；`get_state()` 负责观察，`update_state()` 负责修改，继续运行时沿用同一个会话配置。

---

## 5. 在 Agent 中使用 interrupt：看懂两个 for 循环的执行顺序

前面几节的下一步由我们提前安排，新加入的 [agent中使用interrup.py](agent中使用interrup.py) 则把决定交给了模型。用户要求“先问我在哪里，再查当地天气”，模型可以选择调用 `AskHuman`，也可以调用 `web_search`。我们先认清这些角色，再顺着两个 `for` 走一遍。

### 5.1 AskHuman 是提问协议，ask_human 节点才负责暂停

原代码把 `web_search` 和 `AskHuman` 一起传给 `model.bind_tools()`，模型因而知道有这两个可选工具。`web_search` 是带有具体实现的函数，由 `ToolNode` 执行；`AskHuman` 是 Pydantic 模型，描述的是“提问时需要一个 question 字段”。仅仅绑定它不会弹出输入框，也不会自动向终端读入文字。

真正连接两者的是 `should_continue()`。它看到模型发出了 `AskHuman` 工具调用，就让图进入 `ask_human` 节点；后者调用 `interrupt()`，才真正触发暂停。原代码只检查 `tool_calls[0]`，因此这里先按“模型每次只调用一个工具”来理解流程。

| 位置 | 负责什么 | 在这个例子中的结果 |
| --- | --- | --- |
| `agent` / `call_model` | 把消息交给模型，取得回复 | 发出 `AskHuman` 工具调用 |
| `should_continue` | 根据回复选择下一节点 | 路由到 `ask_human` |
| `ask_human` | 暂停并接收人类回答 | 得到城市，写回工具结果 |
| `action` / `ToolNode` | 执行普通工具 | 调用 `web_search` |

原代码里的 `interrupt("请提供您的位置:")` 使用了固定文案，并没有读取 `AskHuman.question`。这样演示位置收集没问题，但如果希望展示模型实际问的问题，就应当读取 `tool_call["args"]["question"]`。另外，`web_search` 当前直接返回“上海阳光明媚”，它是模拟天气结果，尚未访问搜索服务。

> **小结：** 模型负责提出工具调用，路由负责选节点，`interrupt()` 负责暂停；`AskHuman` 这个名字本身没有暂停程序的魔法。

### 5.2 为什么 interrupt 后，会执行最后一个 for？

**因为第一次 `app.stream()` 在中断后结束了本次事件流，第一个 `for` 随之结束，Python 按顺序执行到第二个 `for`。** 第二个循环又调用了一次 `app.stream()`，这次传入的是 `Command(resume="上海")`，于是图才开始恢复。这两个循环是两次调用，各自消费自己的事件流。

这里同时存在两条执行线：Python 脚本从上往下执行，LangGraph 在一次 `stream()` 调用里调度节点。图停在 `ask_human`，表示这条工作流还有任务没完成；调用方却已经拿回了控制权，可以打印提示、返回 HTTP 响应，也可以马上调用下一次 `stream()`。所以“工作流暂停”和“整个脚本卡住等待输入”是两件事。

把它想成去窗口办业务：材料不全时，工作人员登记进度，让你回去补材料。你的业务仍然处于待处理状态，但窗口这次接待已经结束；等你带着材料再来，才会继续办理。在代码里，两次 `stream()` 就是两次接待，`thread_id` 帮你找到原来的业务记录。

原示例的执行顺序可以按下面这张表来读：

| 顺序 | Python 调用方 | 图内部发生什么 |
| --- | --- | --- |
| 1 | 进入第一个 `for`，迭代第一条事件流 | 接收用户消息，执行 `agent` |
| 2 | 继续接收事件 | 模型请求 `AskHuman`，进入 `ask_human` |
| 3 | 收到中断事件，随后本次事件流结束 | `interrupt()` 发出中断信号，暂停任务并保存可恢复进度 |
| 4 | 第一个 `for` 结束，执行两个循环之间的代码 | 图仍在等待回答，没有自动恢复 |
| 5 | 进入第二个 `for`，迭代新的事件流 | `Command(resume="上海")` 提供恢复值 |
| 6 | 接收恢复后的事件 | `ask_human` 从头执行，`interrupt()` 返回“上海” |
| 7 | 继续接收事件 | 写入工具结果，回到 `agent`，再查天气并生成回复 |
| 8 | 第二个 `for` 结束 | 如果没有新的中断，图到达 `END` |

**你可能会想：没有 `break`，第一个 `for` 怎么会退出？** Python 的 `for` 只要迭代器耗尽就会结束，并不要求一定写 `break`。在这个单分支示例中，LangGraph 处理完中断、输出相关事件后结束当前流，循环自然就退出了；它不会一直阻塞在那里等一个未来的 `resume`。

再往里面看一层：首次执行 `interrupt()` 时，它通过特殊的 `GraphInterrupt` 控制信号把暂停交给 LangGraph 运行时处理。此时 `location = interrupt(...)` 还没有完成赋值，下面构造工具消息的代码也没有执行。调用方正常消费完中断事件后便退出循环，无需自己捕获这个内部控制信号，也不要在节点里用宽泛的异常捕获把它吞掉。

> **小结：** 第一个流结束，Python 才走到第二个循环；第二个循环里的 `Command(resume=...)` 才是恢复的原因。

### 5.3 完整实验：用编号日志观察两次 stream

先把模型调用换成确定的消息生成逻辑，我们就能稳定复现“问城市 → 等回答 → 查天气 → 回复”。下面保留 `MessagesState`、工具调用、`ToolNode` 和中断恢复，只用 `demo_model()` 模拟模型的三次选择，因此不需要 API 密钥、网络或图片渲染。将整段代码保存为 `agent_interrupt_trace.py`，在项目根目录运行 `.venv/bin/python agent_interrupt_trace.py`，特别留意 `[B]` 出现的位置。

~~~python
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt


@tool
def web_search(query: str) -> str:
    """返回演示用的天气信息，不访问网络。"""
    print(f"[工具] 查询：{query}")
    return "上海阳光明媚（模拟数据）"


def demo_model(state: MessagesState):
    """模拟模型选择工具；真实项目中换成绑定工具后的 model.invoke。"""
    last = state["messages"][-1]
    if last.type == "human":
        reply = AIMessage(
            content="",
            tool_calls=[{
                "name": "AskHuman",
                "args": {"question": "请提供您的位置："},
                "id": "ask-location-1",
                "type": "tool_call",
            }],
        )
    elif isinstance(last, ToolMessage) and last.name == "AskHuman":
        reply = AIMessage(
            content="",
            tool_calls=[{
                "name": "web_search",
                "args": {"query": f"{last.content}天气"},
                "id": "search-weather-1",
                "type": "tool_call",
            }],
        )
    else:
        reply = AIMessage(content=f"天气查询结果：{last.content}")
    return {"messages": [reply]}


def should_continue(
    state: MessagesState,
) -> Literal["ask_human", "action", "__end__"]:
    calls = state["messages"][-1].tool_calls
    if not calls:
        return END
    if calls[0]["name"] == "AskHuman":
        return "ask_human"
    return "action"


def ask_human(state: MessagesState):
    call = state["messages"][-1].tool_calls[0]
    print("[节点] ask_human：执行到 interrupt 前")
    location = interrupt(call["args"]["question"])
    print(f"[节点] ask_human：interrupt 已返回 {location}")
    return {"messages": [ToolMessage(
        content=location,
        tool_call_id=call["id"],
        name="AskHuman",
    )]}


def show_event(label, event):
    # 优先识别控制事件，不假定每个事件都包含 messages。
    if "__interrupt__" in event:
        for item in event["__interrupt__"]:
            print(f"{label} 中断问题：{item.value}")
    elif event.get("messages"):
        last = event["messages"][-1]
        calls = getattr(last, "tool_calls", [])
        content = last.content or [call["name"] for call in calls]
        print(f"{label} {last.type}：{content}")


builder = StateGraph(MessagesState)
builder.add_node("agent", demo_model)
builder.add_node("ask_human", ask_human)
builder.add_node("action", ToolNode([web_search]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("ask_human", "agent")
builder.add_edge("action", "agent")
app = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "agent-trace-1"}}

print("[A] 开始第一个 for")
for event in app.stream(
    {"messages": [("user", "先询问我的位置，再查天气")]},
    config,
    stream_mode="values",
):
    show_event("第一轮", event)

snapshot = app.get_state(config)
print("[B] 第一个 for 已结束，下一节点：", snapshot.next)
assert snapshot.next == ("ask_human",)
assert snapshot.values["messages"][-1].tool_calls[0]["name"] == "AskHuman"

print("[C] 开始第二个 for，提交回答：上海")
for event in app.stream(
    Command(resume="上海"), config, stream_mode="values"
):
    show_event("第二轮", event)

snapshot = app.get_state(config)
print("[D] 第二个 for 已结束，下一节点：", snapshot.next)
assert snapshot.next == ()
assert snapshot.values["messages"][-1].content == (
    "天气查询结果：上海阳光明媚（模拟数据）"
)
~~~

代码里的 `demo_model()` 只用于演示模型选择，真正的暂停、保存、恢复和工具执行仍由 LangGraph 完成。固定的工具调用 ID 也只服务于这次独立实验，真实模型调用会返回自己的 ID。当前项目环境为 LangGraph 1.2.10，以上示例使用默认流格式和 `stream_mode="values"`；其他版本若改变事件封装，应按对应版本解析。

运行时的关键顺序如下，省略了部分状态消息：

~~~text
[A] 开始第一个 for
[节点] ask_human：执行到 interrupt 前
第一轮 中断问题：请提供您的位置：
[B] 第一个 for 已结束，下一节点： ('ask_human',)
[C] 开始第二个 for，提交回答：上海
[节点] ask_human：执行到 interrupt 前
[节点] ask_human：interrupt 已返回 上海
[工具] 查询：上海天气
[D] 第二个 for 已结束，下一节点： ()
~~~

请注意，“执行到 interrupt 前”打印了两次，而“interrupt 已返回 上海”只打印一次。第一次节点在赋值完成前暂停；第二次节点从头重跑，框架把恢复值交给对应的 `interrupt()`，函数才能往下走。已经完成的前一个 `agent` 节点不需要在恢复前重新请求模型，等 `ask_human` 返回后，图才沿着边再次进入 `agent`。

**你可能会想：第二个循环开头怎么又打印了 AskHuman 消息？** `stream_mode="values"` 输出的是状态快照，恢复时可能先输出已有消息状态，因此 `messages[-1]` 仍然是暂停前的模型消息。重复打印某条消息不等于模型重新执行了一遍，判断节点是否执行应结合节点日志和状态快照，而不能只数 `pretty_print()` 出现了几次。

> **小结：** `[B]` 证明调用方已经拿回控制权，第二次节点日志证明节点发生了重跑；这两件事合起来，才是完整的中断恢复过程。

### 5.4 为什么恢复后还要构造 ToolMessage？

原代码把 `location` 包装成 `{"type": "tool", "tool_call_id": ..., "content": location}`，它表达的就是一条工具结果消息。模型之前发出了 `AskHuman` 调用，现在需要知道“这个调用得到的结果是上海”；`tool_call_id` 就像回执上的流水号，把结果关联到那次请求。上面的完整示例改用显式的 `ToolMessage`，含义相同，阅读时更容易看清角色。

`MessagesState` 通过消息 reducer 合并更新，所以节点只返回新增的工具消息即可。模型下一次看到的历史包含用户需求、自己的 `AskHuman` 请求，以及对应的“上海”结果，于是可以发出 `web_search` 调用。`ToolNode` 执行搜索、补上搜索结果后，模型再生成最终答案，这才形成完整的工具调用闭环。

**为什么不把“上海”当作新的用户消息传进去？** 在当前图的设计中，模型正在等待一个工具结果，我们要完成的是这个待处理调用。`Command(resume="上海")` 先把值交给中断点，再由节点构造匹配 ID 的工具消息；直接传新的 `messages` 不等价于给这个 `interrupt()` 提供返回值，也可能留下未配对的工具调用。

> **小结：** `resume` 把答案交还节点，`ToolMessage` 把答案交还模型，`tool_call_id` 让两边对上同一次请求。

### 5.5 消息协议在哪里看：不要靠死记字典 key

像下面这样的字典：

~~~python
{
    "role": "tool",
    "name": tool_call["name"],
    "content": result,
    "tool_call_id": tool_call["id"],
}
~~~

不是随便定义的字段集合，而是消息协议的一种写法。它表达的是：“某个工具执行完了，把结果返回给大模型，并说明这个结果对应哪一次工具调用。”可以用四个问题来记忆：

| 字段 | 记忆问题 | 含义 |
| --- | --- | --- |
| `role` | 谁发的？ | 这条消息来自工具 |
| `name` | 哪个工具？ | 工具名称，例如 `weather_search` |
| `content` | 返回了什么？ | 工具执行结果，例如 `晴天` |
| `tool_call_id` | 对应哪次调用？ | 与模型之前生成的工具调用 ID 配对 |

其中 `tool_call_id` 最关键。模型先请求一次工具调用：

~~~python
{
    "id": "call_123",
    "name": "weather_search",
    "args": {"city": "上海"},
}
~~~

工具返回时要带上同一个 ID：

~~~python
{
    "role": "tool",
    "content": "晴天",
    "tool_call_id": "call_123",
}
~~~

这样框架才能知道“晴天”是对哪一次 `weather_search` 请求的回答。实际项目中，与其手写字典，不如使用 LangChain 提供的消息类：

~~~python
from langchain_core.messages import ToolMessage

message = ToolMessage(
    content=result,
    name=tool_call["name"],
    tool_call_id=tool_call["id"],
)
~~~

查协议时可以按三层来找：

1. **LangChain 消息协议**：查看 [Messages](https://python.langchain.com/docs/concepts/messages/)，了解 `HumanMessage`、`AIMessage`、`ToolMessage` 和 `SystemMessage`。
2. **工具调用协议**：查看 [Tools](https://python.langchain.com/docs/concepts/tools/)，了解 `name`、`args`、`id` 和 `tool_calls`。
3. **底层模型协议**：查看 [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)，了解 `role="tool"`、`tool_calls` 和 `tool_call_id`。`ChatOpenAI` 会把这个底层格式封装成 LangChain 对象；使用 DeepSeek 等 OpenAI 兼容服务时，还要以服务商支持的格式为准。

不要只看文档猜结构，也可以直接打印当前版本返回的真实对象：

~~~python
response = model.invoke(state["messages"])

print(type(response))
print(response.model_dump())
print(response.tool_calls)
~~~

工具参数的 JSON Schema 则可以这样查看：

~~~python
print(weather_search.args_schema.model_json_schema())
~~~

这会显示 `city` 的类型和必填信息。学习这类字典时，不必一次背下所有 key；先确认它描述的对象，再用官方文档、类型提示、编辑器补全和 `model_dump()` 核对字段。可以先记住这句：**工具消息就是“角色、工具名、返回内容、调用编号”**。

> **小结：** 消息字段来自 LangChain 和模型接口的约定，不是个人习惯；优先使用 `ToolMessage`，需要确认细节时直接查看文档或打印对象。

### 5.6 真正等待用户输入：在两次 stream 之间接入交互

原示例里没有输入操作，`"上海"` 是程序预先写好的回答，因此两次循环会紧接着执行。要观察“人在中间停留”的效果，可以保留上面完整示例中 `[B]` 之前的代码，把 `[C]` 到 `[D]` 的调用方代码替换成下面这一段。这里是依附于上一份完整示例的替换片段，会一直处理到图不再有待回答的中断。

~~~python
while True:
    snapshot = app.get_state(config)
    pending = [
        item
        for task in snapshot.tasks
        for item in task.interrupts
    ]
    if not pending:
        break
    if len(pending) != 1:
        raise RuntimeError("这个终端示例只处理一个待回答的中断")

    print("需要回答：", pending[0].value)
    answer = input("请输入城市：").strip()
    if not answer:
        print("城市不能为空，请重新输入。")
        continue

    for event in app.stream(
        Command(resume=answer), config, stream_mode="values"
    ):
        show_event("恢复轮", event)

print("当前下一节点：", app.get_state(config).next)
~~~

现在真正阻塞终端的是图外的 `input()`。我们仍让 `interrupt()` 负责图内部的暂停，而把界面输入放在调用方，这样将来换成网页表单，节点本身不用跟着改。终端输入只是这份实验的交互方式，检查到中断以后也可以直接把问题返回给浏览器。

Web 应用通常把这两次 `stream()` 分到两个请求中：第一个请求运行到暂停并返回问题，第二个请求携带用户回答和原 `thread_id` 再恢复。恢复请求还必须访问到原先的检查点数据，仅仅记住 ID 不够；如果服务重启或跨进程处理，就需要共享的持久化 checkpointer。无论用户隔了几秒还是几分钟回答，控制流都没有要求第一次请求一直占着连接等待。

> **小结：** 自动继续来自写死的回答；真正的人机交互，要由图外的终端、网页或回调来收集回答并发起恢复。

### 5.7 回到原文件：读代码时顺手检查这四处

第一处是中断事件的打印。第一个循环对 `messages` 做了存在性判断，所以当前版本返回只有 `__interrupt__` 的事件时，它能走 `else`；第二个循环却直接取 `event["messages"]`。如果恢复后模型又提问、图再次中断，第二个循环就可能因为事件没有 `messages` 而触发 `KeyError`，建议两个循环都使用上面的 `show_event()`。

第二处是“一次只调用一个工具”的假设。原来的路由和 `ask_human` 都只看第一个工具调用，而模型有可能同时返回多个调用。学习阶段可以像完整实验一样控制为单调用；实际应用要么使用模型支持的串行工具调用配置，要么显式设计分发与配对逻辑，不能只处理 `AskHuman` 就忽略同批的其他调用。

第三处是图的可视化。原文件写着“该图生的不对”，其中一个需要检查的原因是 `should_continue()` 没有返回类型或显式路径映射，绘图工具难以推断条件边的范围。完整示例声明了 `Literal["ask_human", "action", "__end__"]`，让可能的目标更明确；这解决的是绘图推断问题，两个 `for` 的执行顺序仍由前面讲过的调用关系决定。

第四处是外部依赖。原文件会调用真实模型，并在开始两个循环之前请求 Mermaid 图片，所以遇到模型配置或绘图网络问题，程序可能根本没有走到中断逻辑。排查时可以先运行本节的离线实验，确认暂停和恢复的顺序，再接回真实模型和可选的图片生成。

> **本节小结：** 人工问答在图中是一条工具调用路径，在调用方则是两次独立的流式调用；把这两个视角分开看，两个 `for` 的衔接就很清楚了。

---

## 6. 按需求选择：动态中断、静态断点和状态查询

把几个 API 放在一起看，会更容易做设计：

| 需求 | 推荐方式 | 关键 API |
| --- | --- | --- |
| 执行到某个业务问题时等待人回答 | 动态中断 | `interrupt()` + `Command(resume=...)` |
| 节点执行前人工检查或修改输入 | 静态断点 | `interrupt_before` + `get_state()` + `update_state()` |
| 查看已经发生过的状态变化 | 检查点历史查询 | `get_state_history()` |
| 根据审批结果跳到不同节点 | 动态路由 | `Command(goto=...)` |

`Command` 在这里扮演两个角色。调用图时，`Command(resume=value)` 是“带着人的回答回来”；节点返回时，`Command(goto="某节点")` 是“我已经决定下一站去哪儿”。如果节点还需要同时改状态，可以写成 `Command(update={"status": "approved"}, goto="next")`。

`get_state_history()` 只是读取历史快照，并不会暂停或重新执行节点。状态编辑则会创建新的检查点，字段更新仍遵循 reducer 规则：普通字段通常覆盖，有追加 reducer 的字段可能追加。`update_state(..., as_node="node_name")` 表示把这次更新视作由指定节点产生，它还会影响后续调度；因此应检查更新后的 `next`，不要把它当作随手填写的标签。

> **小结：** 先按暂停原因选机制，再决定如何恢复；不要把所有人工交互都塞进一个巨大的节点里。

---

## 7. 最佳实践：把暂停流程做得可靠

**第一，给每个会话稳定的 `thread_id`。** Web 请求、任务队列和人工审核回调必须拿到同一个 ID，否则看起来就像“明明暂停了，恢复却找不到状态”。不要把用户姓名直接当作唯一 ID，最好使用订单号、任务号或随机生成的会话标识。

**第二，让中断前的代码可重复执行。** 节点恢复时会从头开始，日志打印、纯计算没有问题；发消息、扣款、写入外部系统则要做幂等控制。一个实用办法是给副作用生成业务唯一键，重复执行时检测到已完成就跳过。

**第三，给恢复值做边界校验。** `interrupt()` 收到的值来自人或外部系统，不能直接相信类型和内容。审批场景可以只接受明确的布尔值，表单场景要同时检查类型、范围和业务权限，失败后再次 `interrupt()` 给出具体提示。

**第四，生产环境不要依赖 `MemorySaver`。** 它非常适合演示，但进程重启、部署扩容后，内存里的检查点就不在了。上线前应选用持久化存储，测试断点恢复、并发更新、历史清理和敏感字段保护。

**第五，把 UI 数据和内部状态分开。** 传给 `interrupt()` 的字典可以包含问题、展示文本和选项，但不要把 API 密钥、完整授权头或不必要的个人信息放进状态。检查点可能长期保存，状态设计得越克制，后续维护越轻松。

**第六，调用方按中断事件组织交互。** 第一轮结束后先检查是否真的有待回答的中断，再收集答案并恢复，不要假定模型一定会走 `AskHuman`。两轮流都应处理 `__interrupt__`，这样模型再次提问时，界面才能继续接住问题。

如果你想继续练习，可以先运行仓库里的原始文件，再把审批示例的 `True/False` 换成一个简单网页按钮；接着给年龄表单增加最大年龄限制；最后在 `step_2` 前加入一个“人工润色摘要”的状态编辑页面。每做一步，都观察 `get_state(config).values` 和 `next`，这比只盯着终端日志更容易理解图到底走到了哪里。

> **小结：** 中断能力本身不复杂，难的是把会话标识、幂等副作用、输入校验和持久化一起设计好。先用 `MemorySaver` 把流程跑通，再逐项补齐生产要求。

---

## 参考示例

- [interrupt_demo.py](interrupt_demo.py)：人工审批与 `Command(goto=...)`
- [interrrput_demo2.py](interrrput_demo2.py)：循环输入校验与多次恢复
- [如何编辑图状态.py](如何编辑图状态.py)：`interrupt_before`、状态快照和 `update_state()`
- [agent中使用interrup.py](agent中使用interrup.py)：Agent 人工问答、工具结果配对与两次 `stream()` 调用
