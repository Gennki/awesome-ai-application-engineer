# LangGraph 子图：把工作流装进节点，再把状态和控制权接回来

前面的 [初识 Graph](../../base/09_初识Graph：State、Node与Edge.md) 介绍了 State、Node 和 Edge，[中断与编辑状态](../interr/11_LangGraph中断与编辑状态.md) 则说明了工作流怎样暂停和恢复。这一篇继续解决一个组织代码的问题：**如果某一步本身就是一套完整流程，能不能把这套流程作为一个节点，接进更大的图？**

比如一个问答流程里，“检索资料”可以包含问题改写、搜索和筛选三个步骤。父图只需要知道输入问题、拿到资料；子图负责组织内部步骤。等需要换检索策略时，我们就可以集中修改这一小块流程。

当前目录只有两个 Python 文件，但一共包含三个值得分开理解的场景：

| 示例 | 代码当前状态 | 学习重点 |
| --- | --- | --- |
| [Subgraphs.py](Subgraphs.py) 第一部分 | 整段示例被注释 | 父子图共享 `foo`，直接把编译后的子图作为节点 |
| [Subgraphs.py](Subgraphs.py) 第二部分 | 当前实际运行的代码 | 父图使用 `foo`，子图使用 `bar`、`baz`，由包装节点转换状态 |
| [示例7_如何将控制流和状态更新与命令结合.py](示例7_如何将控制流和状态更新与命令结合.py) | 当前实际运行的代码 | 子图通过 `Command.PARENT` 更新父图状态，并选择父图的下一节点 |

本文的输出已在项目虚拟环境中的 **LangGraph 1.2.10** 核对。示例均为本地字符串处理，不调用大模型，不需要 API 密钥。流式输出沿用仓库里的 `graph.stream()` 调用形式，不与其他流式接口或版本的事件格式混用。

## 知识点目录

1. [子图是什么：从父图看是节点，从内部看是工作流](#1-子图是什么从父图看是节点从内部看是工作流)
2. [共享状态键：直接把子图作为节点](#2-共享状态键直接把子图作为节点)
3. [不同状态结构：在节点里调用子图并转换数据](#3-不同状态结构在节点里调用子图并转换数据)
4. [流式输出：分清父图事件与子图事件](#4-流式输出分清父图事件与子图事件)
5. [Command.PARENT：从子图跳到父图节点](#5-commandparent从子图跳到父图节点)
6. [Reducer：为什么结果是 bcab 或 bcbc](#6-reducer为什么结果是-bcab-或-bcbc)
7. [常见疑问与排查顺序](#7-常见疑问与排查顺序)
8. [运行与练习：把三个场景逐一验证](#8-运行与练习把三个场景逐一验证)
9. [按需求选择子图的接入方式](#9-按需求选择子图的接入方式)

---

## 1. 子图是什么：从父图看是节点，从内部看是工作流

普通节点通常对应一个 Python 函数。子图则先把多个节点编译成可运行的图，再接入父图。用当前示例中的名字表示，结构是：

```text
父图：START → node_1 → node_2 → 本轮结束
                         │
                         └─ 子图：START → subgraph_node_1 → subgraph_node_2
```

这里的 `node_2` 是父图中的节点名，`subgraph_node_1` 和 `subgraph_node_2` 是子图内部的节点名。父图不需要为子图内部的每一步重新添加边。

先区分两个变量：

```python
subgraph_builder = StateGraph(SubgraphState)
# 在 builder 上添加节点和边……
subgraph = subgraph_builder.compile()
```

`subgraph_builder` 用来描述结构；`subgraph` 是编译后的可运行对象，能够调用 `invoke()`、`stream()`，也能作为父图节点的实现。

阅读子图代码时，可以始终带着三个问题：

1. 父图怎样把数据交给子图？
2. 子图的哪些结果会写回父图？
3. 子图完成后，由谁决定下一节点？

前两个问题决定状态的接口，第三个问题决定控制流。下面先看状态，再看跳转。

> **小结：** 子图把一组处理步骤封装成可组合的工作流。对父图来说，它仍然占据一个节点位置。

---

## 2. 共享状态键：直接把子图作为节点

### 2.1 共享的是字段约定，不要求两个 Schema 完全相同

`Subgraphs.py` 第一部分的父子图都定义了 `foo`，但子图还多了一个 `bar`：

```python
class ParentState(TypedDict):
    foo: str


class SubgraphState(TypedDict):
    foo: str  # 与父图共享
    bar: str  # 仅供子图内部使用
```

父图可以把 `foo` 交给子图，子图结束后也能把新的 `foo` 写回父图。`bar` 没有出现在这个父图的状态结构里，因此不会成为父图最终输出的字段。

这里的“共享”描述的是状态通道的输入和输出约定。写节点时仍然应该返回状态更新，不要把父子图想成可以随意原地修改的同一个 Python 字典。

### 2.2 完整示例：共享 foo，保留内部 bar

原文件的这一部分被注释了。下面将其整理为可以独立运行的版本，最后使用 `invoke()` 直接查看最终状态：

```python
from typing import TypedDict

from langgraph.graph import START, StateGraph


class SubgraphState(TypedDict):
    foo: str
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


class ParentState(TypedDict):
    foo: str


def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

print(graph.invoke({"foo": "foo"}))
```

输出为：

```text
{'foo': 'hi! foobar'}
```

关键的一行是：

```python
builder.add_node("node_2", subgraph)
```

第二个参数直接传入编译后的子图，不需要额外写一个 `node_2()` 函数。

### 2.3 沿着状态走一遍

| 步骤 | 所在图 | 读取或产生的数据 |
| --- | --- | --- |
| 输入 | 父图 | `foo = "foo"` |
| `node_1` | 父图 | 把 `foo` 更新为 `"hi! foo"` |
| 进入 `node_2` | 父图 → 子图 | 子图接收共享字段 `foo = "hi! foo"` |
| `subgraph_node_1` | 子图 | 新增内部字段 `bar = "bar"` |
| `subgraph_node_2` | 子图 | 读取 `foo` 和 `bar`，返回 `foo = "hi! foobar"` |
| `node_2` 完成 | 子图 → 父图 | 父图得到 `foo = "hi! foobar"` |

这里的 `foo: str` 没有指定 reducer。在本例的顺序执行中，新写入的 `foo` 会替换原值。字符串的拼接来自节点函数中的 `+`，不是框架自动追加。

`bar` 也不是在声明 `TypedDict` 时自动生成的。它由 `subgraph_node_1` 写入，后面的 `subgraph_node_2` 才能读取。若跳过第一步，直接访问尚未写入的 `state["bar"]`，就可能出现 `KeyError`。

> **小结：** 共享键负责父子图之间的数据交换，私有键负责子图内部的中间结果。两个图不必使用完全相同的状态结构。

---

## 3. 不同状态结构：在节点里调用子图并转换数据

### 3.1 父图的 foo 不会自动变成子图的 bar

`Subgraphs.py` 当前实际运行的是第二种方式：父图仍然使用 `foo`，子图改为使用 `bar` 和 `baz`。

```text
父图：foo
       │ 输入转换：foo → bar
       ▼
子图：bar、baz
       │ 输出转换：bar → foo
       ▼
父图：foo
```

双方没有共享字段，框架无法猜测 `foo` 应该对应 `bar`。所以父图需要一个包装节点，明确约定输入和输出的转换方式。

### 3.2 完整示例：包装节点负责两次转换

下面保留原文件的处理逻辑，并在最后打印最终状态：

```python
from typing import TypedDict

from langgraph.graph import START, StateGraph


class SubgraphState(TypedDict):
    bar: str
    baz: str


def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}


def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


class ParentState(TypedDict):
    foo: str


def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}


def node_2(state: ParentState):
    response = subgraph.invoke({"bar": state["foo"]})
    return {"foo": response["bar"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

print(graph.invoke({"foo": "foo"}))
```

输出为：

```text
{'foo': 'hi! foobaz'}
```

留意注册节点时第二个参数的变化：

```python
builder.add_node("node_2", node_2)
```

这次传入的是包装函数 `node_2`。函数内部才调用 `subgraph.invoke()`，因此能够在调用前后分别做转换。

### 3.3 response 是子图的结果，return 才是父图的更新

顺着 `node_2()` 的两行代码看：

```text
父图当前状态：{"foo": "hi! foo"}
    ↓ 输入转换
子图收到：    {"bar": "hi! foo"}
    ↓ subgraph_node_1 写入 baz
子图中间状态：{"bar": "hi! foo", "baz": "baz"}
    ↓ subgraph_node_2 更新 bar
子图最终结果：{"bar": "hi! foobaz", "baz": "baz"}
    ↓ 输出转换
父节点返回：  {"foo": "hi! foobaz"}
```

调用子图取得 `response`，不等于已经完成父图状态更新。包装节点必须把需要的结果转换成父图字段并返回。

在这里直接 `return response` 并不能完成 `foo` 的更新，因为它只有 `bar` 和 `baz`。同样，直接把父图的 `state` 传给子图也没有完成输入转换，子图需要的 `bar` 没有被提供。

即使两个图存在同名字段，也可以使用包装函数来筛选、重命名或整理输入输出。选择这种方式的关键是“是否需要转换”，而不只是“有没有同名键”。

> **小结：** 包装节点定义了父子图之间的接口：调用前准备子图输入，调用后把结果转换为父图更新。

---

## 4. 流式输出：分清父图事件与子图事件

### 4.1 默认只观察父图节点的更新

以下片段接在第 3 节的图定义之后运行：

```python
for chunk in graph.stream({"foo": "foo"}, stream_mode="updates"):
    print(chunk)
```

输出为：

```text
{'node_1': {'foo': 'hi! foo'}}
{'node_2': {'foo': 'hi! foobaz'}}
```

这个视角把 `node_2` 看作一个整体。子图内部的两个节点仍然执行了，只是没有单独出现在父图的事件流中。

仓库里省略了 `stream_mode`，当前这些 `StateGraph` 示例的默认输出就是 `updates`。在笔记中显式写出来，有助于区分“节点更新”与“完整状态”。

### 4.2 subgraphs=True 带来命名空间和子图事件

```python
for namespace, update in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    subgraphs=True,
):
    print(namespace, update)
```

每个事件可以拆成 `(namespace, update)`。下面把每次运行都会变化的任务标识统一写成 `<任务ID>`：

```text
() {'node_1': {'foo': 'hi! foo'}}
('node_2:<任务ID>',) {'subgraph_node_1': {'baz': 'baz'}}
('node_2:<任务ID>',) {'subgraph_node_2': {'bar': 'hi! foobaz'}}
() {'node_2': {'foo': 'hi! foobaz'}}
```

| 内容 | 怎样理解 |
| --- | --- |
| `()` | 事件来自根层，也就是当前父图 |
| `('node_2:<任务ID>',)` | 事件来自父图 `node_2` 这次任务所调用的子图 |
| `subgraph_node_1` / `subgraph_node_2` | 真正产生更新的子图内部节点 |
| `{'baz': 'baz'}` | 这个节点提交的更新，不是子图的完整状态 |
| 最后一条 `node_2` 更新 | 包装函数把子图输出转换成父图的 `foo` |

任务标识用于区分执行实例，不能硬编码。一个元素的元组末尾有逗号，所以会显示为 `('node_2:...',)`。

**为什么子图内部用了 `invoke()`，外层还能看到它的事件？** 在这个同步包装节点里，子图是在父图执行上下文中被调用的；外层开启子图事件后，内部更新也能被收集。`invoke()` 仍然给包装函数返回最终结果，这与外层观察执行过程并不冲突。

### 4.3 开启子图事件不会改变父图输出结构

流里看到了 `baz`，不代表父图多出了 `baz` 字段。当前父图的最终输出仍然是：

```python
{"foo": "hi! foobaz"}
```

“子图私有状态”表示它不直接属于父图的状态接口，不代表它在调试输出里不可见。把子图流式事件交给界面或日志时，应先决定哪些字段可以展示。

另外，`Subgraphs.py` 第一部分注释中的两个 `for` 分别调用了一次 `graph.stream()`。如果把它们启用，会完整运行同一张图两次；第二个循环不是继续消费第一次运行留下的事件。当前示例没有配置 Checkpointer，两次也不会自动接续历史状态。

> **小结：** `subgraphs=True` 改变观察范围；Schema 和节点返回值决定状态怎样传递。事件中看见某个字段，不等于它已经进入父图状态。

---

## 5. Command.PARENT：从子图跳到父图节点

### 5.1 这次由子图决定父图下一步去哪儿

前面的子图只负责产出结果。[示例7](示例7_如何将控制流和状态更新与命令结合.py) 则让子图在完成处理时，顺便决定父图接下来执行 `node_b` 还是 `node_c`。

```text
父图 START → subgraph
                │
                └─ 子图 START → node_a
                                   │
                          Command.PARENT
                           ├─ value == "a" → 父图 node_b → 结束
                           └─ value == "b" → 父图 node_c → 结束
```

这里的箭头表示运行路径；原代码没有为两条返回父图的路径写 `add_edge()`，跳转目标由 `Command` 提供。

### 5.2 拆开 Command 的三个参数

原节点的关键逻辑如下：

```python
def node_a(state: State):
    print("Called A")
    value = random.choice(["a", "b"])
    print(value)
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"

    return Command(
        update={"foo": value},
        goto=goto,
        graph=Command.PARENT,
    )
```

| 参数 | 本例的含义 |
| --- | --- |
| `update={"foo": value}` | 向目标图提交 `foo` 的更新，合并方式由目标状态的 reducer 决定 |
| `goto=goto` | 指定接下来执行的节点名：`node_b` 或 `node_c` |
| `graph=Command.PARENT` | 将这条命令交给最近一层父图处理 |

`goto` 是节点名称，不是函数对象。`node_b` 和 `node_c` 注册在父图中，所以要指定 `Command.PARENT`。省略 `graph` 时，命令在当前图中解释，不能指望它自动去父图寻找节点。

如果存在“祖父图 → 父图 → 子图”三层结构，`Command.PARENT` 指向直接包含当前子图的那一层，不会自动跳到最外层。

### 5.3 子图里只注册 A，父图里注册 B 和 C

原文件先构建子图：

```python
subgraph = (
    StateGraph(State)
    .add_node(node_a)
    .add_edge(START, "node_a")
    .compile()
)
```

再定义父图的两个候选节点并挂载子图：

```python
def node_b(state: State):
    print("Called B")
    return {"foo": "b"}


def node_c(state: State):
    print("Called C")
    return {"foo": "c"}


builder = StateGraph(State)
builder.add_edge(START, "subgraph")
builder.add_node("subgraph", subgraph)
builder.add_node(node_b)
builder.add_node(node_c)
graph = builder.compile()
```

虽然没有 `builder.add_edge("subgraph", "node_b")`，B 仍然可以执行，因为 `node_a` 返回的命令会动态安排父图后续节点。

这与上篇中用于恢复的 `Command(resume=...)` 也不同：这里的命令由节点返回，负责更新和路由；恢复命令由调用方传给 `invoke()` 或 `stream()`。本例没有 `interrupt()`，也没有等待用户输入。

> **小结：** `update` 描述数据变化，`goto` 描述下一节点，`graph` 描述命令作用在哪一层。读这三个参数时要一起看。

---

## 6. Reducer：为什么结果是 bcab 或 bcbc

### 6.1 这里的 foo 从覆盖变成了拼接

示例7的状态声明与前两个例子不同：

```python
import operator
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    foo: Annotated[str, operator.add]
```

`operator.add` 对字符串执行拼接，相当于：

```text
新的 foo = 已有 foo + 本次提交的 foo
```

当子图通过 `Command.PARENT` 更新父子图共享的键时，应在父图的该字段上定义 reducer，明确更新怎样合并。本例父子图使用同一个 `State`，所以双方都有这个规则。

这不是说“所有子图都必须使用 `operator.add`”。前两个示例中的普通字符串字段使用覆盖规则就足够；这里需要结合父图命令更新和业务语义选择 reducer。

### 6.2 随机选 a：进入 B，得到 bcab

原文件输入是：

```python
graph.invoke({"foo": "bc"})
```

当 `random.choice()` 返回 `"a"`：

| 时刻 | 父图 foo | 原因 |
| --- | --- | --- |
| 接收输入 | `"bc"` | 初始输入 |
| 处理来自子图的命令更新 | `"bca"` | `"bc" + "a"` |
| 执行 `node_b` 后 | `"bcab"` | `"bca" + "b"` |

终端输出为：

```text
Called A
a
Called B
{'foo': 'bcab'}
```

### 6.3 随机选 b：进入 C，得到 bcbc

当 `random.choice()` 返回 `"b"`，节点选择的是 `node_c`：

```text
输入：             "bc"
Command 更新：     "bc" + "b" = "bcb"
node_c 返回更新：  "bcb" + "c" = "bcbc"
```

对应输出：

```text
Called A
b
Called C
{'foo': 'bcbc'}
```

初始字符串 `"bc"` 只是业务数据，不表示 B、C 已经执行过；随机值 `"b"` 也不表示要执行 B，真正的目标由 `if/else` 中的 `goto` 决定。

### 6.4 有追加 reducer 时，提交本次增量

原文件写的是：

```python
update={"foo": value}
```

不要为了“保留旧值”改成：

```python
# 错误示范：本例已有追加 reducer，会重复拼接旧值。
update={"foo": state["foo"] + value}
```

当旧值是 `"bc"`，随机值是 `"a"` 时，这种写法提交了 `"bca"`；父图又通过 reducer 与自己的 `"bc"` 合并，就变成 `"bcbca"`，再执行 B 得到 `"bcbcab"`。

与第 2 节对比就更容易理解：默认覆盖时，节点自己构造要保存的完整新字符串；追加 reducer 下，当前示例只应提交本次新增的字符。不能只看字段类型是 `str`，还要看它的更新规则。

> **小结：** 本例结果可以直接按“初始值 + 子图命令增量 + 父图目标节点增量”推导。不要把原值包含在增量中再次提交。

---

## 7. 常见疑问与排查顺序

### 7.1 为什么看不到第一种方式的输出？

`Subgraphs.py` 第一部分被注释，直接运行文件只会执行第二部分。当前输出以 `baz` 和 `hi! foobaz` 为特征；看到这些内容是正常的。

建议独立运行第 2 节整理后的代码。原文件的两部分复用了 `SubgraphState`、`subgraph`、`builder` 和 `graph` 等名称；如果同时启用，后面会重新绑定这些名称，阅读调试变量时要确认正在看哪一个示例。

### 7.2 为什么没写 END 也能结束？

当前三个流程都是有限的执行路径。末尾节点执行后，没有新的任务被安排，本轮运行就完成了。子图里的 `START` 和父图里的 `START` 也各自属于自己的图，并不是一条共同的入口。

为了强调终点，可以在前两个示例中显式添加末尾到 `END` 的边；这不是代码能够运行的必要补丁。示例7则要先理解 `Command` 已经承担了动态路由。

### 7.3 能不能同时保留固定边和 Command 跳转？

不要把 `Command(goto=...)` 理解成会取消所有已有连接的 Python `return`。它提供动态路由，普通静态边仍然可能安排执行。例如一个节点在同一图中已有到 B 的固定边，又返回去 C 的命令，B 和 C 都可能执行。

因此，表达“二选一”时应当有清晰的路由来源。排查多执行节点、重复更新时，先检查是否既写了固定出边，又从相关节点返回了动态跳转命令。

### 7.4 Literal 导入了，为什么代码里没有用？

示例7导入了 `Literal`，但当前 `node_a()` 没有返回类型注解。可以用下面的函数签名表达可能的目标：

```python
def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    # 函数体仍使用上文的 Command 逻辑。
    ...
```

这是一段签名示意，不要用省略号替换原节点实现。返回类型能够帮助类型检查和图结构展示理解可能的目标；运行时到底选择 B 还是 C，仍然由实际返回的 `goto` 决定。父图中的节点也仍需显式注册。

### 7.5 子图是不是天然具有记忆？

本目录三个场景的父图都只是 `builder.compile()`，没有配置 Checkpointer，所以不要把 `subgraphs=True`、子图变量复用或多次 `invoke()` 当成开启了会话记忆。

如果后续需要子图内暂停与恢复，可以继续结合 [中断与编辑状态](../interr/11_LangGraph中断与编辑状态.md) 学习 Checkpointer 和 `thread_id`。是否跨调用保留子图内部状态，还要明确配置子图的持久化方式，不能从本例的流式输出推导出来。

### 7.6 状态或跳转出错，先检查什么？

| 现象 | 优先检查 |
| --- | --- |
| 读取 `bar` 或 `baz` 时报 `KeyError` | 输入转换是否提供了字段，生产该字段的前置节点是否执行 |
| 子图有结果，父图 `foo` 没变化 | 包装节点是否返回了父图字段，而不是直接返回子图字典 |
| 只能看到 `node_2`，看不到内部节点 | 是否在外层 `stream()` 开启了 `subgraphs=True` |
| 输出不能按字典方式读取 | 当前事件是否为 `(namespace, update)`，是否选了其他流式模式 |
| 跳转目标不执行 | 节点注册在哪一层，`goto` 名称和 `graph` 是否正确 |
| 结果重复包含输入前缀 | 追加 reducer 下是否又返回了含旧值的完整字符串 |
| 每次运行结果不一样 | `random.choice()` 是否走了不同分支 |

> **小结：** 先确认正在运行哪个示例，再检查状态接口、字段合并规则和目标图层级，最后看事件输出。

---

## 8. 运行与练习：把三个场景逐一验证

### 8.1 从项目根目录运行现有文件

项目已经安装依赖时，可以直接执行：

```bash
.venv/bin/python src/agent/lang_graph/adv/subgraphs/Subgraphs.py
.venv/bin/python src/agent/lang_graph/adv/subgraphs/示例7_如何将控制流和状态更新与命令结合.py
```

第一条命令输出第 4 节中的四条父子图事件，任务 ID 每次不同。第二条命令输出 A 加上一个随机分支，最终 `foo` 为 `bcab` 或 `bcbc`。

### 8.2 不修改源码，固定随机结果检查两个分支

为了避免反复运行仍然只碰到同一分支，可以临时替换 `random.choice()`。下面命令在仓库根目录运行，分别强制返回 `a` 和 `b`，并断言最终结果：

```bash
.venv/bin/python - <<'PY'
import contextlib
import io
import runpy
from pathlib import Path
from unittest.mock import patch

path = Path(
    "src/agent/lang_graph/adv/subgraphs/"
    "示例7_如何将控制流和状态更新与命令结合.py"
)

for choice, expected in (("a", "bcab"), ("b", "bcbc")):
    with patch("random.choice", return_value=choice):
        # 原脚本在加载时会执行一次 invoke；收起那次演示的打印。
        with contextlib.redirect_stdout(io.StringIO()):
            namespace = runpy.run_path(str(path))
        result = namespace["graph"].invoke({"foo": "bc"})
    assert result == {"foo": expected}, result
    print(f"选择 {choice}，最终状态：{result}")
PY
```

这个实验不调用外部服务，也不会修改原文件。因为父图没有 Checkpointer，脚本加载时的那次执行不会把 `foo` 累积到随后断言的调用中。

### 8.3 三个小练习

1. 运行第 2 节的完整示例，把子图产生的 `bar` 改成 `"!"`，先推导结果，再检查父图是否只返回 `foo`。
2. 运行第 3 节的完整示例，把包装节点的返回值改成 `{"foo": response["bar"].upper()}`，观察父图输出与子图内部流式输出的区别。
3. 用第 8.2 节固定两个分支，观察 `Called B` 和 `Called C`，再对照 reducer 的拼接步骤检查结果。

每次只改变一个条件，更容易区分变化来自节点逻辑、状态转换还是路由选择。

---

## 9. 按需求选择子图的接入方式

| 需求 | 采用的方式 | 当前代码中的位置 |
| --- | --- | --- |
| 共享字段已经满足输入输出约定 | 直接把编译后的子图传给 `add_node()` | `Subgraphs.py` 第一部分 |
| 字段不同，或需要筛选、转换结果 | 在包装节点中调用 `subgraph.invoke()` | `Subgraphs.py` 第二部分 |
| 子图需要决定父图下一节点 | 返回 `Command(update=..., goto=..., graph=Command.PARENT)` | 示例7 |
| 希望观察子图内部每一步 | 外层调用 `stream(..., subgraphs=True)` | `Subgraphs.py` 的流式调用 |

前两种方式解决“数据怎样交接”，`Command.PARENT` 解决“控制权交到哪里”，流式参数解决“怎样观察运行”。它们关注的不是同一个维度，不必把它们理解成四种互斥方案。

可以用这张清单复习本目录：

```text
StateGraph → 添加节点和边 → compile → 可运行子图
共享键 → 直接作为父图节点
不同字段 / 需要转换 → 包装节点处理输入输出
subgraphs=True → 观察带命名空间的子图事件
Command.PARENT → 向最近父图提交更新与跳转
Reducer → 决定已有值与本次更新怎样合并
```

> **小结：** 先设计父子图的数据接口，再决定下一步由谁选择，最后用流式事件验证真实执行过程。

---

## 参考示例与延伸阅读

- [Subgraphs.py](Subgraphs.py)：共享键与状态转换两种接入方式。
- [示例7_如何将控制流和状态更新与命令结合.py](示例7_如何将控制流和状态更新与命令结合.py)：父图跳转与字符串 reducer。
- [初识 Graph](../../base/09_初识Graph：State、Node与Edge.md)：补充 State、Schema 和 reducer 基础。
- [图可视化与进阶示例](../10_图可视化：Graphviz与Mermaid.md)：对照图结构理解节点与路由。
- [中断与编辑状态](../interr/11_LangGraph中断与编辑状态.md)：区分 `Command(resume=...)` 与节点返回的路由命令。

官方文档中的 Subgraphs、Graph API overview 和 Use the graph API 可用于核对子图接口、父图命令和 reducer 规则。本文的运行结果以仓库代码和上述本地版本为准；阅读官方示例时，注意它们可能使用不同的流式接口。

```text
https://docs.langchain.com/oss/python/langgraph/use-subgraphs
https://docs.langchain.com/oss/python/langgraph/graph-api
https://docs.langchain.com/oss/python/langgraph/use-graph-api
```
