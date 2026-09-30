# 初识 Graph：理解 State、Node 与 Edge

上一篇 [LangGraph 导读](../../08_LangGraph导读.md) 建立了整体印象。这一篇从一张最小聊天图出发，逐步回答：数据怎样进入图、节点怎样处理数据，以及多个节点为什么可以看到不同的字段。

可以按下面的顺序，边读笔记边查看代码：

| 示例 | 重点 | 对应章节 |
|---|---|---|
| [初识Graph.py](初识Graph.py) | State、Node、Edge 如何一起工作 | 第 1～9 节 |
| [深入Graph.py](深入Graph.py) | 用 Pydantic 定义 State，与 TypedDict 有什么区别 | 第 10 节 |
| [Schema.py](Schema.py) | 分开图的输入、内部状态和输出 | 第 11 节 |
| [Multiple_schemas_a.py](Multiple_schemas_a.py) | 不同节点读取不同字段，传递内部数据 | 第 12 节 |
| [Multiple_schemas_b.py](Multiple_schemas_b.py) | 把输入、输出和节点 Schema 组合起来 | 第 13 节 |
| [Reducers.py](Reducers.py) | 用 reducer 决定新旧状态怎样合并 | 第 14 节 |
| [Messages_Reducer.py](Messages_Reducer.py) | 用列表拼接 reducer 累积对话消息 | 第 15 节 |
| [Messages_Customisze_Reducer.py](Messages_Customisze_Reducer.py) | 比较 `add_messages`、`MessagesState` 和自定义 reducer | 第 16 节 |
| [步骤序列.py](步骤序列.py) | 用 `add_sequence` 快速声明固定步骤 | 第 18 节 |
| [条件边.py](条件边.py) | 根据 State 动态选择一条或多条路径 | 第 19 节 |

阅读时始终带着三个问题：**流程带着什么数据？每一步做什么？做完以后去哪里？** 它们分别对应 State、Node 和 Edge。

## 1. Graph 究竟是什么

Graph 是“图”，Graphs 是它的复数。在 LangGraph 中，图表示处理步骤及其连接关系，可以先理解成一张能够执行的流程图。

当前示例的任务很简单：接收用户输入，调用一次聊天模型，然后结束本次处理。

```text
START ──→ chatbot ──→ END
             │
             └─ 读取 messages → 调用模型 → 返回 messages 更新
```

这张图中，`chatbot` 是真正执行工作的节点；两条箭头规定执行顺序；`START` 和 `END` 分别标记流程入口与出口。

普通 Python 函数也能完成这个任务。采用图的表达方式，是为了把“每一步的工作”和“步骤之间如何连接”分别定义。以后加入输入处理、检索资料、条件分支时，就可以在这张结构上扩展。

LangGraph 的图可以包含分支和循环，不要求所有流程都是从左到右的一条直线。不过，本篇示例只有一个业务节点，没有图内循环。

| 概念 | 回答的问题 | 当前示例 |
|---|---|---|
| Graph | 整个流程怎样组织？ | `START → chatbot → END` |
| State | 流程当前携带什么数据？ | 包含 `messages` 的字典 |
| Node | 这一步要做什么？ | `chatbot` 函数调用模型 |
| Edge | 做完以后执行哪一步？ | 入口到 `chatbot`，再到出口 |

## 2. State：先约定流程中的数据结构

原代码先定义状态：

```python
class State(TypedDict):
    messages: list[AnyMessage]
```

### 2.1 State 不只是输入参数

状态描述图执行过程中使用的数据。它既可以包含用户输入，也可以包含节点产生的中间结果和最终结果。

当前图只有 `messages` 一个字段。将来做 RAG，可以增加问题、检索文档、回答等字段，让不同节点读取和更新各自需要的内容。

这里的 `State` 是状态结构声明，也称为状态 Schema。Schema 可以理解成“数据的结构约定”；一次调用传入的字典，才是这次运行的具体数据：

```python
{"messages": [{"role": "user", "content": "你好"}]}
```

`State`、`messages` 和 `chatbot` 都是示例选用的名称。普通 `StateGraph` 不要求状态必须叫 `State`，也不要求必须有 `messages` 字段；字段应与节点实际读写的数据保持一致。

### 2.2 TypedDict 与 AnyMessage 分别负责什么

`TypedDict` 声明字典有哪些键，以及各个值预期是什么类型。这里约定 `messages` 是消息对象列表。

`AnyMessage` 是 LangChain 中多种消息类型的联合类型，例如 `HumanMessage`、`AIMessage`、`SystemMessage` 和 `ToolMessage`。它表达的是“允许不同角色的消息对象”，并不表示任意 Python 对象。

`TypedDict` 主要为编辑器和类型检查工具提供信息，不能单凭它获得严格的运行时数据校验，也不会自动把字符串转换成 `AIMessage`。

这解释了原代码中的一个细节：它声明 `list[AnyMessage]`，调用时却传入消息字典，节点返回时又放入 `message.content`。这种写法可能运行成功，但实际数据与类型声明并不完全一致。

输入的消息字典能够交给 `llm.invoke()`，是聊天模型接口支持这种消息输入格式；不能据此认为 `TypedDict` 已把字典转换成消息对象。

### 2.3 当前 messages 的更新规则是替换

原示例中的定义是：

```python
messages: list[AnyMessage]
```

它没有配置 reducer（状态更新的合并函数）。在本例的顺序执行中，节点写入新的 `messages` 时，会用新值替换该字段原来的值。

```text
更新前：{"messages": [用户输入]}
节点返回：{"messages": [回复内容]}
更新后：{"messages": [回复内容]}
```

**列表类型不会让 LangGraph 自动追加元素。** 是否追加，由字段的更新规则决定。

原文件注释写着“使用 Annotated 添加元数据”，但实际声明没有使用 `Annotated`，应以实际代码为准。上一篇中的下面这种写法才配置了消息合并规则：

```python
messages: Annotated[list[AnyMessage], add_messages]
```

其中 `add_messages` 通常追加新消息，也可以按相同消息 id 更新已有消息。第 14～17 节会系统比较默认覆盖、列表拼接和消息专用 reducer。

## 3. Node：读取状态，完成工作，返回更新

原代码的业务节点是：

```python
def chatbot(state: State):
    llm = getModel()
    print(f"聊天机器人节点接收状态：{state['messages']}")
    message = llm.invoke(state["messages"])
    print(f"大模型答复：{message.content}")
    return {"messages": [message.content]}
```

这里将打印语句中的内层引号写成单引号，便于阅读，处理逻辑与原文件一致。

### 3.1 输入是当前状态

在这个 TypedDict 示例中，`state` 是图在执行到这个节点时交给它的状态字典。后文的 Pydantic 示例中，节点收到的则是状态模型实例。

本例只有一个节点，因此它拿到的是本轮输入。如果前面还有其他节点，它可能拿到那些节点更新后的状态。节点通过 `state["messages"]` 取出消息列表，再把它交给模型。

`getModel()` 位于函数内部，因此每次执行这个节点都会调用模型工厂函数。

### 3.2 工作内容由函数决定

```python
message = llm.invoke(state["messages"])
```

这一步才真正调用聊天模型。图并不会因为节点名叫 `chatbot` 就自动具备聊天能力，能力来自函数内部的实现。

同样，节点也可以做普通字符串处理、数据库查询或文档检索。理解 Node 时，先把它当成图中的一个处理步骤即可。

### 3.3 返回值是状态更新

```python
return {"messages": [message.content]}
```

节点把更新交给 LangGraph，框架再根据字段规则更新状态。本例写入 `messages`，因此替换原消息列表。

节点不需要返回所有状态字段。如果状态还有 `question` 和 `answer`，只返回 `{"answer": "回答内容"}` 就表示更新 `answer`，未写入的字段保留原值。后文的多个 Schema 示例会反复用到这种“只返回本步更新”的写法。

原代码的两个 `print()` 只是终端输出，不会把数据写入 State；决定状态更新的是 `return`。

### 3.4 message 与 message.content 要分清

| 写法 | 保存什么 | 后续如何取内容 |
|---|---|---|
| `[message]` | 模型返回的消息对象，通常为 `AIMessage` | `messages[-1].content` |
| `[message.content]` | 消息内容，纯文本回复时为字符串 | `messages[-1]` |

原代码使用第二种写法。对于简单文本输出可以观察到结果，但会丢失消息角色及其他消息元数据；部分模型的 `content` 也可能是内容块列表，而不只是字符串。

若要让返回值与 `list[AnyMessage]` 的声明一致，更合适的节点返回是：

```python
return {"messages": [message]}
```

这只是对状态表示的改进建议。它保留消息对象，但仍然遵循默认替换规则，不会因此自动累积聊天历史。

## 4. 注册 Node：名字和函数各有什么用

定义函数后，原代码将它加入图：

```python
graph_builder.add_node("chatbot", chatbot)
```

两个参数看起来相同，职责却不同：

- `"chatbot"` 是节点在图中的名字，后续连边时用它定位节点。
- `chatbot` 是函数对象，图运行到这里时调用它。

也可以注册为 `add_node("reply", chatbot)`，这时连边必须使用 `"reply"`。

这里传入函数本身，没有写 `chatbot()`。定义函数、注册节点都不会立即调用模型，真正执行发生在后面的 `graph.invoke()` 中。

## 5. Edge：描述控制流

原代码添加两条普通边：

```python
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```

第一条表示启动本次图执行后进入 `chatbot`；第二条表示 `chatbot` 正常完成、状态更新后，结束本次图执行。

`START` 和 `END` 是特殊标记，不需要自己编写或注册对应函数。`END` 也不表示退出整个 Python 程序：图结束后，控制权返回调用它的代码。

边主要规定“下一步执行谁”。节点返回的更新先写入图的状态，下游节点再从更新后的状态中读取自己需要的字段。因此，沿着边传递的数据不一定等于上一个函数的返回字典。

执行顺序由边决定，单纯先注册 A、再注册 B，并不能表达“A 完成后执行 B”。

以后遇到条件边，可以在运行时依据状态选择去向；当前示例只使用固定边，所以每次路径都是 `START → chatbot → END`。

## 6. 从构建器到可运行的图

将前面的内容连起来，构建过程如下：

```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

`StateGraph(State)` 创建构建器，并告诉它使用哪种状态结构。代码执行到这一行前，`State` 必须已经定义。

`graph_builder` 用来声明节点与边；`compile()` 检查图结构并生成可执行图；`graph.invoke(input_state)` 才启动一次执行。编译不会提前询问模型，也不意味着检查了所有业务输入是否正确。

可以把这几个对象放在一起区分：

| 对象 | 当前职责 |
|---|---|
| `State` | 声明数据结构 |
| `graph_builder` | 组织处理步骤和连接关系 |
| `graph` | 接收输入并按图执行 |
| `state` | 某次节点执行时拿到的具体数据 |

## 7. 跟踪一次完整调用

假设用户输入“你好”，模型恰好回复“你好！有什么可以帮你？”。以下回复仅用来演示数据变化。

```python
result = graph.invoke({
    "messages": [{"role": "user", "content": "你好"}]
})
```

按照原文件的节点返回写法，这次执行过程是：

| 阶段 | 发生的事情 | messages 的内容 |
|---|---|---|
| 接收输入 | 图得到本轮状态数据 | `[{"role": "user", "content": "你好"}]` |
| START | 根据入口边调度 `chatbot` | 保持输入值 |
| chatbot 内部 | 模型读取消息，生成回复 | 此时回复先保存在局部变量 `message` 中 |
| 节点返回后 | 应用 `{"messages": [message.content]}` | `["你好！有什么可以帮你？"]` |
| END | 结束本次执行并返回最终状态 | `["你好！有什么可以帮你？"]` |

因此 `result` 的示意值为：

```python
{"messages": ["你好！有什么可以帮你？"]}
```

这个示例没有单独指定输出 Schema，因此图返回状态 Schema 中的最终字段。原文件没有将 `graph.invoke(...)` 的返回值赋给变量，所以终端看到的回答来自节点里的 `print()`。后文会进一步限制图对外返回的字段。

按原代码取结果，应使用 `result["messages"][-1]`。如果先把节点返回改成 `[message]`，才相应使用 `result["messages"][-1].content`。

## 8. while True 是交互循环，聊天历史需要另行管理

原文件外围的逻辑可以简化为：

```python
while True:
    user_input = input("User:")
    if user_input.lower() in ["quit", "exit", "q"]:
        break
    graph.invoke({"messages": [{"role": "user", "content": user_input}]})
```

每次用户输入，都会发起一次新的图调用：

```text
第 1 次输入 → START → chatbot → END → 回到 input()
第 2 次输入 → START → chatbot → END → 回到 input()
第 3 次输入 → START → chatbot → END → 回到 input()
```

这段代码复用了编译后的图结构，但没有把上一轮最终状态传入下一轮，也没有配置 checkpointer。每次发送给模型的只有本轮用户消息。

因此，先说“我叫张三”，再问“我叫什么名字”，模型没有从这段程序获得上一轮姓名。即使偶然猜中，也不能用来证明记忆生效。

与上一篇对比时，要把三个问题分开：State 定义数据结构；reducer 决定状态怎样更新；历史传递或 checkpointer 解决跨次调用怎样延续数据。它们承担不同职责。

## 9. 一个不调用模型的完整练习

为了只观察图和状态，可以单独运行下面的示例。这里用普通字符串字段，让节点处理规则完全确定。

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class DemoState(TypedDict):
    text: str


def add_prefix(state: DemoState):
    return {"text": "处理结果：" + state["text"]}


def add_suffix(state: DemoState):
    return {"text": state["text"] + "！"}


builder = StateGraph(DemoState)
builder.add_node("prefix", add_prefix)
builder.add_node("suffix", add_suffix)
builder.add_edge(START, "prefix")
builder.add_edge("prefix", "suffix")
builder.add_edge("suffix", END)
demo = builder.compile()

print(demo.invoke({"text": "你好"}))
# {'text': '处理结果：你好！'}
```

这张图的路径是 `START → prefix → suffix → END`。第一个节点更新 `text`，第二个节点读取更新后的 `text`，再次返回更新。

可以让第二个节点改为返回 `{"text": "固定结果"}`，观察它如何替换前一步的值；再改为返回空字典 `{}`，观察没有更新字段时，前一步的状态如何保留。

回到聊天示例时，只有节点内的工作换成了模型调用，State、Node、Edge 的配合方式仍然相同。

## 10. Pydantic State：让状态成为可校验的模型

[深入Graph.py](深入Graph.py) 使用了另一种 State 定义：

```python
from typing import List

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field


class State(BaseModel):
    messages: List[AnyMessage] = Field(default_factory=list)
```

它仍然描述同一个问题：图中的状态有哪些字段、每个字段是什么类型。变化在于，`State` 现在继承 `BaseModel`，因此它是可以创建出来的 Pydantic 实例；`TypedDict` 则主要是一个类型声明。

### 10.1 TypedDict 与 Pydantic 的核心区别

| 比较点 | `TypedDict` | `pydantic.BaseModel` |
|---|---|---|
| 本质 | 对字典结构的类型提示 | 可实例化的状态模型类 |
| 节点访问 | `state["messages"]` | `state.messages` |
| 运行时校验 | 默认不主动校验 | 创建或解析模型时会校验字段 |
| 默认值 | 只声明，不负责创建 | 可用 `Field(default_factory=...)` 提供默认值 |
| 数据转换 | 不负责转换 | Pydantic 会按字段类型尝试解析或拒绝输入 |
| 适合场景 | 轻量、透明、状态像普通字典 | 需要约束、默认值、嵌套模型和明确错误 |

例如，普通 TypedDict 不会因为运行时传入字符串就自动报错：

```python
class TypedState(TypedDict):
    count: int


value = TypedState(count="3")  # 运行时仍然只是一个普通字典
```

Pydantic 则会创建并校验模型实例：

```python
class PydanticState(BaseModel):
    count: int = Field(ge=0)


value = PydanticState(count="3")
assert value.count == 3
PydanticState(count=-1)  # ValidationError
```

默认情况下，Pydantic 允许一些类型转换，例如把字符串 `"3"` 转成整数 `3`；使用严格模式时可以限制这种转换。`Field(ge=0)` 进一步要求数值大于或等于零，所以 `-1` 会触发 `ValidationError`。TypedDict 没有这些运行时行为，但类型检查工具仍能指出类型不匹配。

### 10.2 Pydantic State 在节点里如何访问

`深入Graph.py` 的节点写法是：

```python
def chatbot(state: State):
    llm = getModel()
    print(f"聊天机器人节点接收状态：{state.messages}")
    message = llm.invoke(state.messages)
    return {"messages": [message.content]}
```

因为 `state` 是 `State` 实例，所以使用属性访问 `state.messages`。TypedDict 版本的 `state` 按字典方式访问，因此使用 `state["messages"]`。

写节点时，访问方式需要跟随状态对象的类型：

```text
TypedDict State       -> state["messages"]
Pydantic BaseModel    -> state.messages
```

普通字典使用 `state.messages` 通常会触发 `AttributeError`；普通 BaseModel 实例使用 `state["messages"]` 通常会触发 `TypeError`。先看 State 的定义，就能判断该使用哪种访问方式。

### 10.3 用 `Field(default_factory=list)` 声明默认值

```python
messages: List[AnyMessage] = Field(default_factory=list)
```

它表示创建一个 State 时，如果没有提供 `messages`，就调用 `list()`，为这个实例生成一个新的空列表。Pydantic 也会对这里直接写成 `=[]` 的可变默认值进行复制；`default_factory=list` 更明确地表达了“每个实例各建一个列表”的意图。

这里的默认值只解决“缺少字段时如何构造模型”，不等于消息会自动追加，也不等于跨次 `invoke()` 自动保存历史。消息合并仍由 reducer 决定，跨调用保存仍需要 checkpointer 等机制。

### 10.4 消息字典不一定能被 Pydantic 自动转换

`深入Graph.py` 使用了更明确的输入：

```python
graph.invoke({"messages": [HumanMessage(content=user_input)]})
```

这是因为 `AnyMessage` 需要消息类型信息。`HumanMessage` 已经明确说明这是人类消息。

不要把以下两件事混为一谈：

- 聊天模型的 `invoke()` 通常支持 `{"role": "user", "content": "..."}` 这样的消息字典。
- Pydantic 根据 `list[AnyMessage]` 校验状态时，需要能够识别消息类型；只有 `role`/`content` 的字典不一定满足 LangChain 消息解析要求。

更稳妥的写法是显式构造消息对象：

```python
from langchain_core.messages import HumanMessage

graph.invoke({"messages": [HumanMessage(content="你好")]})
```

在本例没有消息 reducer 的情况下，直接使用 `HumanMessage` 可以让输入符合 `AnyMessage` 的预期。这里讨论的是消息表示；如果以后加入 `add_messages`，还要考虑它对消息格式的转换。

### 10.5 Pydantic 不会替节点设计状态更新规则

Pydantic 负责模型实例的字段校验；节点返回什么、状态如何合并，仍由 LangGraph 处理。

`深入Graph.py` 仍然返回：

```python
return {"messages": [message.content]}
```

这里延续了第 3.4 节的表示问题：State 声明的是 `AnyMessage` 列表，纯文本回复时返回的却是字符串列表。在当前环境中，单节点图直接结束时，这个值会原样成为最终输出；如果再连接一个使用该 Pydantic State 的节点，就会在为下游节点构造状态模型时暴露类型错误。不要假定节点返回值和最终图输出都会自动重新经过 Pydantic 校验。

更一致的返回方式是保留消息对象：

```python
return {"messages": [message]}
```

然后下游节点可以使用 `state.messages[-1].content` 读取文本，同时保留角色、消息 id 和其他元数据。

节点收到 BaseModel 实例，也仍然可以返回字典形式的局部更新。本例普通 `graph.invoke(...)` 的结果是字典；若已改为返回 `[message]`，应从 `result["messages"][-1].content` 取文本。

### 10.6 选择哪一种 State

可以按状态复杂度选择：

- 刚开始学习图、状态字段简单，希望看到普通字典如何流动时，`TypedDict` 更直观。
- 需要字段默认值、数值范围、嵌套结构或在边界处尽早发现错误时，Pydantic 更合适。
- 无论选择哪种方式，都应统一消息表示、明确节点返回值，并单独设计 reducer 和 checkpoint 策略。

选好状态的表示方式后，还需要决定哪些字段接收外部输入、哪些字段对外返回。下面的 Schema 示例解决的就是这个问题。

## 11. Schema.py：把图的输入、内部状态和输出分开

[Schema.py](Schema.py) 展示了另一个问题：一张图内部可以有多个状态字段，但调用者不一定需要传入所有字段，也不一定需要看到所有字段。

```python
class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class OverallState(InputState, OutputState):
    pass
```

`InputState` 描述图的入口字段，`OutputState` 描述图对外返回的字段，`OverallState` 是构建图时指定的主状态 Schema。这里通过多继承合并字段，同时声明了 `question` 和 `answer`。`pass` 表示类中没有额外定义，不代表它没有字段。

这种继承是一种组织字段的方式，不是 LangGraph 的强制要求；也可以直接在 `OverallState(TypedDict)` 中写出这两个字段。

### 11.1 `StateGraph` 的三个 schema 参数

```python
builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState,
)
```

```text
输入字典 → 按 InputState 接收 question
         → 节点更新内部状态：question、answer
         → 按 OutputState 返回 answer
```

- 第一个位置参数 `OverallState` 声明图的主要状态字段；后续示例会通过节点输入 Schema 注册额外内部字段。
- `input_schema=InputState` 限定图的入口形式。
- `output_schema=OutputState` 限定图对外返回哪些字段。

图的内部状态可以比输入和输出更丰富。例如输入只需要 `question`，内部保存问题和中间结果，最后只返回 `answer`。这样调用者只需要关心图的输入和最终结果。

### 11.2 为什么传入 `answer` 也不会改变输出

Schema.py 有两次调用：

```python
graph.invoke({"question": "hi"})
graph.invoke({"question": "hi", "answer": "OK"})
```

两次都会返回：

```python
{"answer": "hello"}
```

按照执行顺序，可以分成三步理解：

1. 入口按照 `InputState` 接收 `question`；额外传入的 `answer="OK"` 被过滤，没有作为本轮输入写入状态。
2. `answer_node` 固定写入 `answer="hello"`。它没有根据问题生成答案，只是在演示状态写入。
3. 输出 schema 是 `OutputState`，最终只保留 `answer`。

这里不能解释成“`OK` 先进入状态，再被 `hello` 覆盖”。仅看最终输出无法区分过滤和覆盖，需要同时看入口 Schema。也不要把过滤字段理解成严格的类型校验：本例使用 TypedDict，不会仅凭 `question: str` 就自动验证所有业务输入。

### 11.3 节点可以只声明自己需要的 State

```python
def answer_node(state: InputState):
    return {"answer": "hello", "question": state["question"]}
```

这里 LangGraph 会从节点参数的类型注解推断节点的输入 Schema，让这个节点读取 `question`。它仍然可以返回 `answer`，因为 `answer` 已经是图中注册的状态字段。节点的读取范围和写入字段不必相同。

返回的 `question` 是一次同值更新。也可以只返回 `{"answer": "hello"}`，已有的 `question` 会保留。它没有出现在最终结果中，是因为输出 Schema 只选取 `answer`，并非它从内部状态中被删除。

虽然 `OverallState` 声明了两个字段，第一次调用只需要传入 `question`；`answer` 由节点随后产生。TypedDict 字段声明不会在运行时自动填满所有字段，尚未写入的字段也不应提前读取。

### 11.4 与前面两种 State 写法的关系

`InputState` 和 `OutputState` 仍然是 `TypedDict`，所以 Schema.py 展示的是“输入/输出边界分离”，不是又一种状态对象类型。

| 写法 | 解决的问题 |
|---|---|
| `TypedDict` / `BaseModel` | 每个状态字段怎么表示、如何校验 |
| `input_schema` | 图接受哪些输入字段 |
| `output_schema` | 图返回哪些字段 |
| `OverallState` | 图的主要状态字段；未显式指定输入输出 Schema 时，也用作默认输入输出 Schema |

可以同时使用 Pydantic 模型作为这些 schema，但不要把 `input_schema` 和 `output_schema` 误认成新的节点或边；它们只定义数据进出边界。

### 11.5 什么时候需要区分输入和输出

小型图中，只使用一个 `State` 通常已经足够。当图对外是一个可复用的子流程，或者内部有较多中间字段时，分开 schema 会更清楚：

```text
外部只提供 question
        ↓
图内部运行 question + 检索结果 + 中间判断
        ↓
对外只返回 answer
```

到这里，我们控制了图与调用者之间的数据边界。接下来把视角移到图内部：每个节点也可以只读取自己需要的字段。

## 12. 多 Schema 示例 A：让节点只读取需要的字段

[Multiple_schemas_a.py](Multiple_schemas_a.py) 的执行路径仍是一条直线：

```text
START → node_1 → node_2 → node_3 → END
```

变化的是节点接收的数据结构。先看三个 Schema：

```python
class OverallState(TypedDict):
    a: str


class Node1Output(TypedDict):
    private_data: str


class Node2Input(TypedDict):
    private_data: str
```

`OverallState` 只有 `a`；另外两个 Schema 都有 `private_data`。两个类使用不同名字，是为了强调一个描述上一步的输出，另一个描述下一步的输入。它们指向的是同名的状态字段，不会生成两份 `private_data`。

### 12.1 从函数签名看节点的职责

下面摘出节点的输入和返回值，省略原文件的打印语句：

```python
def node_1(state: OverallState) -> Node1Output:
    return {"private_data": "set by node_1"}


def node_2(state: Node2Input) -> OverallState:
    return {"a": "set by node_2"}


def node_3(state: OverallState) -> OverallState:
    return {"a": "set by node_3"}
```

`node_1` 读取 `a`，写入 `private_data`；`node_2` 读取 `private_data`，写回 `a`；`node_3` 再读取并更新 `a`。三个节点都返回固定字符串，目的是观察字段流转，并没有实际业务运算。

原文件用 `builder.add_node(node_1)` 注册节点。这是省略节点名字的写法，框架使用函数名 `node_1` 作为节点名，所以连边时仍写 `"node_1"`。

### 12.2 private_data 不在 OverallState 中，为什么能传递

构建器最初接收的是：

```python
builder = StateGraph(OverallState)
```

随后注册 `node_2` 时，LangGraph 从 `state: Node2Input` 得到该节点的输入 Schema，并把其中的 `private_data` 注册为图的状态字段。图编译前已经完成所有节点注册，因此运行 `node_1` 时，这个字段已经可写。

内部用于保存字段值的位置通常称为状态通道（channel）。本例只需把它理解成“图已经认识这个字段，可以保存它的值”。实际状态字段可以来自主 Schema，也可以来自另外注册的节点输入 Schema。

`-> Node1Output` 主要说明函数预期返回什么，**不能只靠返回类型注解来注册新字段或过滤节点输出**。本例能传递 `private_data`，关键在于它通过 `Node2Input` 被图注册。写节点时，应返回图已知字段的更新。

### 12.3 跟着真实输出走一遍

输入是 `{"a": "set at start"}`。每个节点收到的字典和返回的更新如下：

| 步骤 | 节点收到的状态 | 节点返回的更新 |
|---|---|---|
| `node_1` | `{"a": "set at start"}` | `{"private_data": "set by node_1"}` |
| `node_2` | `{"private_data": "set by node_1"}` | `{"a": "set by node_2"}` |
| `node_3` | `{"a": "set by node_2"}` | `{"a": "set by node_3"}` |

最终结果是：

```python
{"a": "set by node_3"}
```

`node_2` 没有收到 `a`，因为它的输入 Schema 只声明了 `private_data`。`node_3` 没有收到 `private_data`，因为它使用 `OverallState`，这里只声明了 `a`。

本例没有单独指定 `output_schema`，默认输出使用 `OverallState`，所以最后只返回 `a`。

### 12.4 “私有字段”应该怎样理解

这里的 private 表示“供内部节点协作使用，不属于默认公开输出”。它并没有在 `node_2` 执行后自动删除，也没有只允许两个特定函数访问的特殊机制。

如果另一个节点也使用包含 `private_data` 的输入 Schema，它同样可以读取该字段。在调试、状态检查或配置了 checkpoint 的场景中，内部字段也可能被观察或保存。

因此，可以把节点输入 Schema 理解成节点读取状态时的一份字段清单。它减少了节点需要关心的数据，但不是安全隔离机制。

## 13. 多 Schema 示例 B：把入口、节点和出口连起来

[Multiple_schemas_b.py](Multiple_schemas_b.py) 将前两个示例合在一起：外部只输入一句话的开头，三个节点逐步拼接，最后只返回拼接结果。

### 13.1 先看各个 Schema 管什么

| Schema | 字段 | 用途 |
|---|---|---|
| `InputState` | `user_input` | 图的入口，也是 `node_1` 的输入 |
| `OverallState` | `foo`、`user_input`、`graph_output` | 主状态，也是 `node_2` 的输入 |
| `PrivateState` | `bar` | 内部中间字段，也是 `node_3` 的输入 |
| `OutputState` | `graph_output` | 图对外返回的结果 |

`foo`、`bar` 是演示中常用的占位名字。这里 `foo` 保存第一次拼接结果，`bar` 保存第二次拼接结果。`PrivateState` 不需要继承 `OverallState`；字段可以通过不同的 Schema 注册到同一张图中。

### 13.2 每个节点读取一部分状态，再写回本步结果

```python
def node_1(state: InputState) -> OverallState:
    return {"foo": state["user_input"] + " name"}


def node_2(state: OverallState) -> PrivateState:
    return {"bar": state["foo"] + " is"}


def node_3(state: PrivateState) -> OutputState:
    return {"graph_output": state["bar"] + " YunFang"}
```

图的执行顺序和 A 一样：`START → node_1 → node_2 → node_3 → END`。数据按下面的方式变化：

```text
user_input = "My"
    ↓ node_1
foo = "My name"
    ↓ node_2
bar = "My name is"
    ↓ node_3
graph_output = "My name is YunFang"
```

这张示意图标出了每步产生的字段。旧字段仍保存在内部状态里，下游节点读取哪些字段，由各自的输入 Schema 决定。

| 步骤 | 节点实际收到的字典 | 写入字段 |
|---|---|---|
| `node_1` | `{"user_input": "My"}` | `foo` |
| `node_2` | `{"foo": "My name", "user_input": "My"}` | `bar` |
| `node_3` | `{"bar": "My name is"}` | `graph_output` |

有两个细节值得停下来观察。首先，`node_2` 能看到最初的 `user_input`，尽管 `node_1` 没有返回它：局部更新不会清空其他字段。其次，`node_2` 没有收到 `graph_output`，因为这个字段还没被写入；类型声明不会凭空创建值。

### 13.3 读取范围不限制写入同一张图的其他字段

`node_2` 的输入是 `OverallState`，但它可以写入 `bar`；`node_3` 的输入只有 `bar`，却能写入 `graph_output`。

这是因为节点输入 Schema 负责选择读取字段，而返回字典负责提交状态更新。`bar` 通过 `node_3` 的 `PrivateState` 输入声明注册，`graph_output` 则已在主 Schema 和输出 Schema 中声明。

同时要区分 LangGraph 的局部更新与 Python 的静态类型约定。原示例将 `node_1` 的返回类型写成 `OverallState`，实际只返回 `foo`，LangGraph 能按局部更新处理；但严格的类型检查工具可能指出缺少另外两个必填键。需要完善静态类型时，可以单独定义只含 `foo` 的更新 TypedDict，作为返回类型。返回注解本身不会补齐缺少的键。

### 13.4 为什么最终只返回 graph_output

构建器指定了输入和输出边界：

```python
builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState,
)
```

因此调用 `graph.invoke({"user_input": "My"})` 后得到：

```python
{"graph_output": "My name is YunFang"}
```

内部存在的 `user_input`、`foo`、`bar` 都不在 `OutputState` 中，所以没有出现在这个返回值里。原文件节点中的 `print()` 仍会显示中间输入；输出 Schema 只决定图的结果字段，不会抑制这些打印。

### 13.5 改成 StateGraph(OverallState) 会怎样

原文件保留了一行注释，适合用来比较：

```python
builder = StateGraph(OverallState)
```

这样会使用 `OverallState` 作为默认输入和输出 Schema。保持节点和边不变，同样输入 `{"user_input": "My"}`，结果变成：

```python
{
    "foo": "My name",
    "user_input": "My",
    "graph_output": "My name is YunFang",
}
```

`bar` 仍然不返回，因为它不在默认输出 `OverallState` 中。每个节点的输入范围仍由节点自己的 Schema 决定，不会因为移除了图级输入输出参数，就让所有节点都收到全部字段。

现在可以把整个过程连成一句话：图先按入口 Schema 接收输入，再按节点输入 Schema 提供当前字段；节点的更新写回图中，最后按输出 Schema 选取结果。

## 14. Reducer：决定同一字段的新旧值怎样合并

前面的示例已经出现过两种状态更新：没有 reducer 时，新值覆盖旧值；配置 reducer 后，框架把旧值和节点返回的新值交给 reducer。Reducer 常译为“归约器”，也有人称“规约器”。

```text
没有 reducer：字段的新值 = 节点返回值

配置 reducer：字段的新值 = reducer(字段旧值, 节点返回值)
```

Reducer 是绑定在**状态字段**上的，不是绑定在某个节点或某条边上。只要图要把更新写入这个字段，就会按照该字段的规则处理。

### 14.1 用 `Annotated` 给字段绑定 reducer

[Reducers.py](Reducers.py) 定义了两个字段：

```python
from operator import add
from typing import Annotated, TypedDict


class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

`Annotated` 的第一个参数 `list[str]` 是字段类型，第二个参数 `add` 是该字段的 reducer。`operator.add(left, right)` 等价于 `left + right`；当左右两边都是列表时，结果是列表拼接。

`foo` 没有 reducer，因此节点更新它时会覆盖旧值。`bar` 使用 `add`，因此节点更新它时会把新列表拼到旧列表后面。源码注释中的“map 高阶函数”不准确，这里使用的是二元加法函数，与 Python 的 `map()` 无关。

### 14.2 Reducer 接收旧值和本次更新

可以把框架内部的动作简化为：

```python
merged_bar = add(current_bar, node_update_bar)
```

如果当前值是 `['My', 'name']`，节点只需要返回本次新增内容：

```python
def example_node(state: State):
    return {"bar": ["OK"]}
```

框架应用 reducer 后得到：

```python
["My", "name"] + ["OK"]
# ["My", "name", "OK"]
```

Reducer 的两个输入和返回值都应符合字段的使用方式。它应返回合并结果，而不是在不清楚影响范围的情况下直接修改共享对象。

### 14.3 为什么 Reducers.py 的旧内容出现了两次

当前示例节点返回的是：

```python
return {"bar": state["bar"] + ["OK"]}
```

节点先手动把旧状态拼进去，得到 `['My', 'name', 'OK']`；随后 reducer 又执行一次 `旧值 + 节点返回值`：

```text
旧值          = ["My", "name"]
节点返回值    = ["My", "name", "OK"]
reducer 结果  = ["My", "name", "My", "name", "OK"]
```

这正是运行文件时的最终输出。它很好地说明了 reducer 示例最容易犯的错误：**配置累积型 reducer 后，节点通常只返回增量，不要再次带上旧值。**

如果希望得到 `['My', 'name', 'OK']`，将节点返回改为 `{"bar": ["OK"]}` 即可。`foo` 没有被节点更新，所以仍保留输入值 `1`。

### 14.4 为什么需要 reducer

Reducer 让状态合并规则集中在 Schema 中。每个节点只描述“本次产生了什么”，不需要重复编写“先取旧值，再拼接新值”的逻辑。

当多个节点可能更新同一字段时，LangGraph 也需要明确的合并规则。此时 reducer 是否满足业务语义会直接影响结果。例如列表拼接适合保留所有新增项，而计数器可能使用整数加法；“取最大值”“按主键去重”等规则则需要自定义函数。

## 15. 用普通列表 reducer 累积对话消息

[Messages_Reducer.py](Messages_Reducer.py) 把相同思路应用到消息列表：

```python
import operator
from typing import Annotated, Sequence, TypedDict


class ChatState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]
```

`operator.add` 对消息序列执行拼接。两个节点都只返回本次新增的消息：

```python
return {"messages": [HumanMessage(content=user_input)]}
return {"messages": [response]}
```

假设第一次调用前只有系统消息，一轮图执行的数据变化是：

```text
[SystemMessage]
    + [HumanMessage]
    + [AIMessage]
= [SystemMessage, HumanMessage, AIMessage]
```

这种分工是正确的：节点产生增量，reducer 负责累积。

### 15.1 历史为什么能跨多次 `invoke()` 保留

示例在外层循环中执行：

```python
result = conversation.invoke(state)
state = result
```

每次 `invoke()` 仍是一次独立的图运行。历史能够进入下一次运行，是因为程序把上一次返回的完整状态保存到变量 `state`，随后再次传给图。这是**手动传递状态**，不是 reducer 自动把数据存到了多次调用之外，也没有使用 checkpointer。

可以把两种职责分开理解：

```text
reducer：一次状态更新发生时，新值怎样与旧值合并
外层 state = result：下一次调用从哪里得到上一轮状态
checkpointer：由框架按 thread_id 保存和恢复跨调用状态
```

### 15.2 截取最近六条消息不会修改完整历史

```python
recent_history = state["messages"][-6:]
response = model.invoke(recent_history)
```

这里只把最近六条消息交给模型，State 中原有的完整 `messages` 并没有被裁剪。下一轮调用仍会携带所有历史，内存占用也会继续增长。

如果目的是控制长期状态大小，就需要显式返回裁剪后的更新、设计覆盖型字段，或使用消息删除和摘要机制。只对局部变量切片不能修改图状态。

### 15.3 当前“退出”写法的实际行为

示例在节点中写了：

```python
return END
```

`END` 是连边时使用的特殊目标，不是普通节点允许返回的状态更新。节点通常应返回字典，或者使用 LangGraph 支持的路由/命令类型。在当前依赖版本中直接返回 `END` 会触发 `InvalidUpdateError`；外层的异常捕获会结束程序，因此表面看起来像正常退出，实际是通过异常中止。

在学习到条件边之前，可以先在图外读取并判断退出命令；之后再用状态中的退出标记配合条件边，或者使用支持跳转的 `Command`。因此，后面的“检查 HumanMessage 是否为退出”在当前代码路径上不会真正处理到这条退出消息。

## 16. 消息专用 reducer：`add_messages`

普通 `operator.add` 能拼接列表，但不了解 LangChain 消息的 id、类型和删除语义。LangGraph 提供了消息专用 reducer：

```python
from langgraph.graph import add_messages


class BuiltInReducerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    extra_field: int
```

`add_messages` 的常用行为包括：

- 新消息 id 不存在时，追加消息；
- 新旧消息 id 相同时，用新消息更新原位置的消息；
- 将支持的消息字典转换成 LangChain 消息对象；
- 配合 `RemoveMessage(id=...)` 按 id 删除已有消息。

例如，两条消息的 id 都是 `msg-1`：

```python
old = AIMessage(content="旧回答", id="msg-1")
new = AIMessage(content="新回答", id="msg-1")
```

`operator.add` 会保留两条消息；`add_messages` 会得到一条内容为“新回答”的消息。这是聊天状态通常更适合 `add_messages` 的原因。

### 16.1 直接使用预构建的 `MessagesState`

如果状态主要围绕对话消息展开，可以继承预构建状态：

```python
from langgraph.graph import MessagesState


class PreBuildState(MessagesState):
    extra_field: int
```

`MessagesState` 已经声明了带 `add_messages` reducer 的 `messages` 字段。继承后只需增加自己的业务字段，避免重复写消息字段定义。

`extra_field` 没有配置 reducer，因此它被更新时仍使用默认覆盖规则。一份 State 中可以为不同字段配置不同的合并方式。

### 16.2 自定义 reducer 的签名

[Messages_Customisze_Reducer.py](Messages_Customisze_Reducer.py) 当前启用的是自定义函数：

```python
def customize_add(left, right):
    return left + right


class CustomizeReducerState(TypedDict):
    messages: Annotated[list[AnyMessage], customize_add]
    extra_field: int
```

自定义 reducer 接收当前值 `left` 和本次更新 `right`，返回合并后的值。这个 `customize_add` 与 `operator.add` 的当前效果相同，都是普通列表拼接；它没有 `add_messages` 的消息 id 更新、消息转换和删除能力。

自定义 reducer 的价值在于加入真实的业务规则。例如，可以按业务主键去重、限制列表长度或合并统计值。定义时要保证它对所有可能输入都有稳定结果，并考虑多个更新以不同组合顺序合并时是否仍符合预期。

### 16.3 三种消息合并方式怎样选择

| 写法 | 合并行为 | 适用场景 |
|---|---|---|
| 无 reducer | 新列表覆盖旧列表 | 每次只保留最新一批消息 |
| `operator.add` / 当前 `customize_add` | 直接执行列表拼接 | 数据已是规范消息对象，只需要简单追加 |
| `add_messages` / `MessagesState` | 按消息语义追加、更新、转换或删除 | 常规 LangGraph 对话状态 |

对于聊天图，通常优先考虑 `add_messages` 或 `MessagesState`。只有业务确实需要不同合并规则时，再编写自定义 reducer。

## 17. Reducer、历史传递和持久化不要混为一谈

学习消息 State 时，这三个概念很容易混在一起：

| 机制 | 解决的问题 | 当前示例中的体现 |
|---|---|---|
| Reducer | 一次更新怎样合并进当前 State | `operator.add`、`add_messages`、`customize_add` |
| 手动传递状态 | 下一次调用怎样拿到上次结果 | `state = result` 后再次 `invoke(state)` |
| Checkpointer | 框架怎样按会话保存和恢复状态 | 这三个新示例没有配置 |

因此，只给 `messages` 添加 reducer，不会让两个完全独立、且没有传递状态的 `invoke()` 自动共享历史。反过来，即使使用 checkpointer，如果字段没有合适的 reducer，每次新消息更新仍可能覆盖旧消息。

可以用一句执行规则串起这一部分：**节点返回本次增量，reducer 负责合并当前状态；要跨调用保留状态，还需手动传回结果或配置 checkpointer。**

## 18. 步骤序列：用 `add_sequence` 连接固定流程

前面的多个节点示例一直手动写边：

```python
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
```

[步骤序列.py](步骤序列.py) 展示了更紧凑的写法：

```python
graph_builder = (
    StateGraph(State)
    .add_sequence([step_1, step_2, step_3])
)
graph_builder.add_edge(START, "step_1")
```

`add_sequence()` 会按列表顺序注册节点，并在相邻节点之间创建普通边。三个函数形成的图是：

```text
START → step_1 → step_2 → step_3
```

它适合“步骤固定、每一步只去下一步”的线性流程。它不会替节点执行函数，也不会自动把列表元素当成状态字段；真正运行仍发生在 `compile()` 之后的 `invoke()`。

### 18.1 按执行过程看状态如何变化

示例定义了：

```python
class State(TypedDict):
    value_1: str
    value_2: int


def step_1(state: State):
    return {"value_1": "a"}


def step_2(state: State):
    return {"value_1": f"{state['value_1']} + b"}


def step_3(state: State):
    return {"value_2": 10}
```

调用时只传入 `{"value_1": "c"}`，实际运行过程是：

| 步骤 | 节点收到的相关状态 | 节点返回的更新 | 更新后的状态 |
|---|---|---|---|
| 初始 | `value_1="c"` | 无 | `value_1="c"` |
| `step_1` | `value_1="c"` | `value_1="a"` | `value_1="a"` |
| `step_2` | `value_1="a"` | `value_1="a + b"` | `value_1="a + b"` |
| `step_3` | `value_1="a + b"` | `value_2=10` | `value_1="a + b", value_2=10` |

最终输出是：

```python
{"value_1": "a + b", "value_2": 10}
```

这里没有 reducer，`step_1` 和 `step_2` 对 `value_1` 的更新都是覆盖。`step_3` 只返回 `value_2`，因此已有的 `value_1` 保留。节点返回局部更新这一规则，与前面多 Schema 和 reducer 示例中的规则相同。

### 18.2 类型声明不等于每次都自动补齐字段

`State` 写了 `value_1` 和 `value_2`，但第一次 `invoke()` 只传了 `value_1`，示例仍能运行。这说明 `TypedDict` 本身只是类型提示，不能把缺少的 `value_2` 自动填成默认值。

在本例中，`value_2` 在 `step_3` 执行前没有被读取，因此缺失没有造成问题。若把 `step_2` 改成读取 `state["value_2"]`，就要在入口传入它，或先由某个节点写入它。需要默认值和运行时校验时，应考虑 Pydantic State 或显式的输入初始化节点。

### 18.3 `add_sequence` 和手动连边如何选择

`add_sequence` 只适合图结构本身是线性的情况。遇到以下需求时，应回到显式 `add_node()` 和 `add_edge()`：

- 某一步需要根据 State 选择不同下游节点；
- 一个节点需要同时分发给多个节点；
- 流程中存在循环、提前结束或人工确认；
- 节点名称需要与函数名不同。

序列构建器降低了固定流程的书写量，但没有改变 Node、State、Edge 和 reducer 的基本规则。

## 19. 条件边：让 State 决定下一步

固定边在构建时就确定路径，例如 `a → b`。条件边会在图运行到某个节点后调用一个路由函数，再根据路由函数的返回值选择下游节点。[条件边.py](条件边.py) 的图可以画成：

```text
                 ┌→ b ─┐
START → a ─ route┤     ├→ e → END
                 └→ c ─┘
                   或
                 ┌→ c ─┐
                 └→ d ─┘
```

### 19.1 先看固定部分

示例的 State 是：

```python
class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    which: str
```

`aggregate` 使用列表拼接 reducer，所以每个节点只返回本次新增的字母。`which` 没有 reducer，用来保存路由条件。

节点 `a`、`b`、`c`、`d`、`e` 分别追加对应字母：

```python
def c(state: State):
    return {"aggregate": ["C"]}
```

图的入口固定是 `a`：

```python
builder.add_edge(START, "a")
```

变化发生在 `a` 执行之后。

### 19.2 路由函数返回“去哪些节点”

```python
def route_bc_or_cd(state: State):
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]
```

路由函数读取当前 State，并返回目标节点名列表。`which="cd"` 时同时选择 `c` 和 `d`；其他值选择 `b` 和 `c`。

这和 Python 的 `if` 有相似之处，但路由函数不负责直接调用 `c()` 或 `d()`，只负责给图返回路径。真正调度仍由 LangGraph 执行。

### 19.3 注册条件边和 `path_map`

```python
intermediates = ["b", "c", "d"]
builder.add_conditional_edges(
    "a",
    route_bc_or_cd,
    path_map=intermediates,
)
```

第一个参数 `"a"` 是从哪个节点出发；第二个参数是路由函数；`path_map` 告诉图路由函数可能返回哪些目标节点。这里返回值本身就是节点名，所以列表直接列出这些名字即可。

更复杂的写法可以让路由函数返回业务标签，再用字典把标签映射到节点名。例如返回 `"need_review"`，映射到 `"review_node"`。学习条件边时，先把 `path_map` 理解成“路由返回值到节点的允许映射”。

如果路由函数返回列表，LangGraph 会把它理解为多条路径，而不是把列表当成一个节点名。返回单个节点名则只走一条路径。

### 19.4 多条路径如何汇合

示例为每个中间节点都连接到 `e`：

```python
for node in intermediates:
    builder.add_edge(node, "e")
builder.add_edge("e", END)
```

当 `which="cd"` 时，路径是：

```text
a → c ─┐
       ├→ e → END
a → d ─┘
```

`aggregate` 的变化是：

```text
[] → ["A"] → ["A", "C"] 和 ["A", "D"] → ["A", "C", "D"] → ["A", "C", "D", "E"]
```

`c` 和 `d` 都完成后，`e` 才会继续执行，因此 `e` 能看到两个分支写入后的聚合结果。分支执行顺序和并发细节不应被当作业务排序依据；如果结果顺序很重要，应在数据中记录顺序或在汇合节点显式排序。

### 19.5 `which="bc"` 时会怎样

如果初始状态改为：

```python
{"aggregate": [], "which": "bc"}
```

路由函数返回 `b、c`，执行路径变为：

```text
START → a → b ─┐
               ├→ e → END
          c ───┘
```

最终聚合内容是 `A、B、C、E`。路由函数中使用的字符串必须与 `path_map` 允许的目标一致；如果返回未注册的目标，编译或运行时会报错。

### 19.6 条件边和条件节点的区别

路由函数 `route_bc_or_cd` 只负责选择路径，不承担业务处理，也不会写入 State。`b/c/d/e` 才是实际执行工作的节点。

把判断逻辑放进单独的路由函数，能让流程图更容易阅读：节点负责做事，路由负责决定下一步。路由函数也可以根据模型分类、检索结果、重试次数或用户确认状态选择路径。

## 20. 阅读顺序与动手验证

阅读新的图时，可以先看 `StateGraph(...)` 的主状态和输入输出 Schema，再顺着边检查每个节点。对每一步问两个问题：“它现在能读到什么？”“它返回了哪些更新？”把答案列成表，就能复原数据流。

运行原聊天示例时，在项目根目录执行：

```bash
.venv/bin/python -m src.agent.lang_graph.base.初识Graph
```

它使用项目的模型配置，会发起真实模型请求。输入 `q`、`quit` 或 `exit` 可退出交互。

三个 Schema 示例只处理本地字符串，可以分别运行，观察节点输入和最终输出：

```bash
.venv/bin/python -m src.agent.lang_graph.base.Schema
.venv/bin/python -m src.agent.lang_graph.base.Multiple_schemas_a
.venv/bin/python -m src.agent.lang_graph.base.Multiple_schemas_b
```

固定步骤和条件边示例不需要模型配置：

```bash
.venv/bin/python -m src.agent.lang_graph.base.步骤序列
.venv/bin/python -m src.agent.lang_graph.base.条件边
```

修改 `条件边.py` 的 `which` 为 `"bc"`，观察路由从 `c、d` 变成 `b、c`；修改 `步骤序列.py` 中的 `step_2`，观察下游节点收到的状态如何变化。

理解后可以尝试三个小练习，每次只改一处，再对照输出：

1. 在 A 中让 `node_3` 使用 `Node2Input`，保持返回值不变，观察它收到的字段是否变为 `private_data`。
2. 在 B 的 `OutputState` 中增加 `foo: str`，观察最终结果是否同时返回 `foo` 和 `graph_output`。
3. 在 B 中让 `node_1` 返回 `{}`，推测 `node_2` 读取 `state["foo"]` 时为什么会失败，再运行验证。

Reducer 示例可以先运行不调用模型的版本，观察重复数据：

```bash
.venv/bin/python -m src.agent.lang_graph.base.Reducers
```

然后将节点返回改为 `{"bar": ["OK"]}` 再次运行，对比两次结果。两个消息示例会调用真实模型并等待终端输入，应在模型配置完成后运行：

```bash
.venv/bin/python -m src.agent.lang_graph.base.Messages_Reducer
.venv/bin/python -m src.agent.lang_graph.base.Messages_Customisze_Reducer
```

第一轮学习先掌握五件事：节点读取的是状态的一部分，节点返回的是本步更新，reducer 决定更新怎样合并，固定边决定必经顺序，条件边根据 State 选择路径。接下来学习循环和中断时，就可以在这套理解上继续扩展。
