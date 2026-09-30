# LangGraph 持久化：会话状态、跨线程记忆与消息管理

前面的 [LangGraph 导读](../../08_LangGraph导读.md) 用 Checkpointer 实现了多轮对话，[子图笔记](../adv/subgraphs/12_LangGraph子图：状态共享、转换与父图跳转.md) 介绍了父子图怎样交换状态。这一篇把两条线接起来，回答三个问题：**同一个会话怎样接着聊？子图内部执行到哪里，能不能查？换一个会话后，还能不能记住同一个用户？**

保存下来的对话会越来越长，因此还需要回答：**哪些消息发给模型？哪些消息从当前状态中删除？怎样用摘要保留旧对话中的重要信息？** 新增的四个示例从这三个角度补充消息管理。

当前目录的七个 Python 文件都包含会直接执行的顶层代码，建议按下面的顺序阅读：

| 示例 | 当前代码做什么 | 学习重点 |
| --- | --- | --- |
| [示例11_如何为图添加线程级持久性.py](示例11_如何为图添加线程级持久性.py) | 同一 `thread_id` 下连续询问姓名 | `MessagesState`、Checkpointer 与消息历史 |
| [示例12_如何为子图添加现成持久特性.py](示例12_如何为子图添加现成持久特性.py) | 执行父子图，再读取父图和子图状态 | Checkpointer 继承、状态历史与检查点命名空间 |
| [示例13_如何为如图添加跨线程级持久性.py](示例13_如何为如图添加跨线程级持久性.py) | 换 `thread_id`，保持 `user_id`，检索之前保存的信息 | Store、用户命名空间与跨会话记忆 |
| [简单Agent聊天消息管理.py](简单Agent聊天消息管理.py) | 在模型与工具之间循环，并保留同线程消息 | 工具绑定、工具调用消息与完整对话状态 |
| [通过函数修剪消息.py](通过函数修剪消息.py) | 每次只将最后一条消息传给模型 | 模型输入裁剪与持久化状态的区别 |
| [删除消息节点.py](删除消息节点.py) | 回答完成后，超过两条时删除最早两条 | `RemoveMessage`、消息 ID 与删除时机 |
| [如何添加会话历史摘要.py](如何添加会话历史摘要.py) | 超过六条消息时生成摘要、保留最后两条，最后再手动删除一条 | 摘要字段、增量摘要与 `update_state()` |

文件名中的“现成”“为如图”沿用源码名称；本文分别按“子图持久化”和“跨线程持久化”解释。

本文在项目虚拟环境的 **LangGraph 1.2.10、langgraph-checkpoint 4.2.0、langchain-core 1.5.3** 下核对。所有示例都使用真实图运行时；涉及模型的部分使用替身模型，示例13另使用本地向量验证数据流，绘图被临时跳过。未调用真实模型、嵌入服务或绘图服务。因此文中的确定结果来自状态与断言，不把模拟回答当作真实模型效果。原三个示例的验证命令见第 8.2 节，新增四个示例见第 13 节。

## 知识点目录

1. [先区分两种保存：Checkpointer 与 Store](#1-先区分两种保存checkpointer-与-store)
2. [示例11：同一线程怎样保留消息历史](#2-示例11同一线程怎样保留消息历史)
3. [stream：状态流与消息打印不是一回事](#3-stream状态流与消息打印不是一回事)
4. [示例12：父图如何把持久化能力交给子图](#4-示例12父图如何把持久化能力交给子图)
5. [状态查询：从父图历史找到子图快照](#5-状态查询从父图历史找到子图快照)
6. [示例13：用 Store 共享跨线程记忆](#6-示例13用-store-共享跨线程记忆)
7. [容易混淆的边界与排查顺序](#7-容易混淆的边界与排查顺序)
8. [运行与练习：离线验证三个示例](#8-运行与练习离线验证三个示例)
9. [简单 Agent：完整保留模型与工具消息](#9-简单-agent完整保留模型与工具消息)
10. [函数修剪：只改变本次模型输入](#10-函数修剪只改变本次模型输入)
11. [删除节点：通过 RemoveMessage 更新状态](#11-删除节点通过-removemessage-更新状态)
12. [会话摘要：把旧消息压缩到 summary](#12-会话摘要把旧消息压缩到-summary)
13. [新增示例的离线验证与练习](#13-新增示例的离线验证与练习)
14. [按需求选择保存与消息管理方式](#14-按需求选择保存与消息管理方式)

---

## 1. 先区分两种保存：Checkpointer 与 Store

“记住上一轮对话”和“在新会话里记住用户偏好”看起来很像，但本目录用了两套机制：

| 对比项 | Checkpointer | Store |
| --- | --- | --- |
| 保存什么 | 图状态及执行恢复所需的检查点信息 | 应用主动写入的业务记录 |
| 本目录实现 | `MemorySaver()` | `InMemoryStore()` |
| 如何定位 | `thread_id`，以及检查点命名空间、检查点 ID | `namespace` 与记录 `key` |
| 谁决定写入内容 | 图运行时按状态更新保存检查点 | 节点显式调用 `store.put()` |
| 如何使用 | 同线程恢复状态，查询执行历史 | 显式检索，再把需要的信息交给模型 |
| 典型场景 | 连续对话、中断恢复、观察中间步骤 | 跨会话读取用户姓名、偏好等资料 |

可以把示例13的数据放在两个位置理解：

```text
Checkpointer
  thread_id="1" → 第一个会话的 messages
  thread_id="3" → 第二个会话的 messages

Store
  namespace=("memories", "1")
    key=<某个 UUID> → {"data": "你好！remember：我的名字是张三"}
```

两个会话的消息历史不同，但节点可以主动访问同一份用户资料。

这里的“线程”是 LangGraph 的逻辑会话标识，不是 Python 的 `threading.Thread`。所谓短期、长期记忆主要描述信息的用途和作用范围，不是给数据设置了固定的存活天数。

还要单独看存储生命周期：**本目录两种实现都保存在内存中。** `MemorySaver` 在本地版本中是 `InMemorySaver` 的别名；它和 `InMemoryStore` 都不会因为名字里出现“持久化”就自动写入磁盘。对象被重新创建、进程重启后，原有内存数据不会自动恢复。

> **小结：** Checkpointer 保存某个会话的图状态，Store 保存应用选择的业务记忆；是否跨重启保留，要看实际采用的存储后端。

---

## 2. 示例11：同一线程怎样保留消息历史

### 2.1 图结构很小，关键在状态与配置

示例11只有一个业务节点：

```text
START → call_model → 本轮结束
```

原文件没有显式添加到 `END` 的边。当前简单图执行完唯一节点后没有后续任务，本轮就结束。

```python
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
```

三部分各有职责：

1. `MessagesState` 提供 `messages` 字段及其 `add_messages` 合并规则。
2. `compile(checkpointer=memory)` 让运行时保存、加载图状态。
3. `thread_id="1"` 让两次调用定位到同一个会话。

`add_messages` 通常追加新消息，遇到相同消息 ID 时则更新对应消息。源码返回单个 `AIMessage` 对象，当前 reducer 可以接收它；写成 `{"messages": [response]}` 也可以。不要据此推断任意列表字段都能接收单个对象，关键在于这里采用的 reducer。

### 2.2 两次调用分别给模型传了什么

原代码先发送“你好，我是张三”，再发送“我的名字是什么”。第二次只需要提交新问题：

```python
input_message = {"role": "user", "content": "我的名字是什么"}
graph.invoke({"messages": [input_message]}, config)
```

下面是两轮正常完成时的消息变化，`AI1`、`AI2` 表示两次模型回答：

| 时机 | `messages` |
| --- | --- |
| 第一轮节点收到状态 | `[Human1]` |
| 第一轮结束并保存 | `[Human1, AI1]` |
| 第二轮节点收到状态 | `[Human1, AI1, Human2]` |
| 第二轮结束并保存 | `[Human1, AI1, Human2, AI2]` |

第二轮仍然会执行 `call_model`。记忆来自恢复的 State 被重新传给 `model.invoke()`，不是模型服务自动记住了上一次 HTTP 请求，也不是第一次调用一直没有结束。

每轮只交新输入即可。若把没有稳定消息 ID 的旧历史反复提交，可能把旧消息重复追加。

### 2.3 完整实验：不用模型，也能验证线程隔离

下面是根据示例11整理的独立实验。节点返回它实际看到的消息数量，这样就不用根据自然语言回答猜测记忆是否生效：

```python
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


def call_model(state: MessagesState):
    return {
        "messages": AIMessage(content=f"本次看到 {len(state['messages'])} 条消息")
    }


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}
for text in ("你好，我是张三", "我的名字是什么"):
    result = graph.invoke({"messages": [("user", text)]}, config)
    print(result["messages"][-1].content)

other_config = {"configurable": {"thread_id": "2"}}
result = graph.invoke({"messages": [("user", "我的名字是什么")]}, other_config)
print(result["messages"][-1].content)
print(len(graph.get_state(config).values["messages"]))
```

输出是：

```text
本次看到 1 条消息
本次看到 3 条消息
本次看到 1 条消息
4
```

原线程保留四条消息，新线程只收到自己的新问题。相同的 `thread_id` 还需要配合能访问到原数据的 Checkpointer；每次请求都新建一个 `MemorySaver()`，不会自动找回前一个对象中的记录。

> **小结：** Reducer 决定消息怎样合并，Checkpointer 决定状态怎样保存，`thread_id` 决定读写哪一份状态。

---

## 3. stream：状态流与消息打印不是一回事

示例11和13使用：

```python
for chunk in graph.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
```

`values` 返回当时的完整状态；`[-1]` 只是从该状态里取最后一条消息来打印。对于本目录的单节点图，通常先看到包含新用户输入的状态，再看到节点回答后的状态，因此终端会同时打印用户消息和 AI 消息。

这不是逐 Token 输出。节点内部仍然调用 `model.invoke()`，循环观察的是图状态变化；`pretty_print()` 也只负责显示，不负责保存记忆。

示例12没有指定 `stream_mode`，沿用当前 `StateGraph` 的默认 `updates` 输出。加上 `subgraphs=True` 后，每项变成：

```python
(namespace, update)
```

例如：

```text
((), {"node_1": {"foo": "hi! foo"}})
(("node_2:<任务ID>",), {"subgraph_node_1": {"bar": "bar"}})
```

原文件写 `for _, chunk in ...`，把事件的命名空间丢弃了，只打印更新。调试时可以改成 `for namespace, chunk in ...` 同时观察两部分。

| 调用方式 | 本目录中用于观察什么 |
| --- | --- |
| `invoke()` | 本次运行返回的最终状态 |
| `stream(..., stream_mode="values")` | 执行过程中的完整状态 |
| `stream(..., subgraphs=True)` | 默认更新模式下，带父子图命名空间的节点更新 |
| `get_state(config)` | 已保存的某个状态快照，不主动执行节点 |

`subgraphs=True` 控制流式事件是否包含子图，不会代替 `compile(checkpointer=...)` 开启持久化。

> **小结：** 先看流式模式，再看每个事件的数据结构。终端打印多少内容，与底层保存多少状态是两件事。

---

## 4. 示例12：父图如何把持久化能力交给子图

### 4.1 沿着 foo 和 bar 还原执行过程

结构与前面的共享状态键子图相同：

```text
父图：START → node_1 → node_2 → 本轮结束
                         │
                         └─ 子图：START → subgraph_node_1 → subgraph_node_2

父图字段：foo
子图字段：foo、bar
```

输入 `{"foo": "foo"}` 之后：

| 步骤 | 更新 | 说明 |
| --- | --- | --- |
| 父图 `node_1` | `foo = "hi! foo"` | 拼接前缀 |
| 子图 `subgraph_node_1` | `bar = "bar"` | 写入子图内部字段 |
| 子图 `subgraph_node_2` | `foo = "hi! foobar"` | 拼接 `foo` 与 `bar` |
| 父图 `node_2` 完成 | `foo = "hi! foobar"` | 共享字段写回父图 |

最终父图状态为 `{"foo": "hi! foobar"}`，子图状态为 `{"foo": "hi! foobar", "bar": "bar"}`。父图没有声明 `bar`，所以不会因为启用持久化就多出这个输出字段。

### 4.2 为什么子图没有单独传 MemorySaver

原文件的关键连接是：

```python
subgraph = subgraph_builder.compile()
builder.add_node("node_2", subgraph)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

子图默认编译后直接作为父图节点，执行时可以继承父图的 Checkpointer。父图和子图的检查点通过不同命名空间区分，子图内部步骤也能被记录。

但是，这不等于示例已经实现“子图跨多次调用自动累积私有历史”。当前子图使用默认的每次调用作用域；如果需求是子 Agent 在同一线程里跨调用积累自己的历史，需要另行了解子图 `compile(checkpointer=True)` 的配置。该模式仍需要父图提供 Checkpointer，本目录没有演示它。

同样，给独立的 `subgraph.invoke()` 随意传一个 `thread_id`，也不等于正在使用父图的运行环境。这里的继承发生在父图执行已接入的子图时。

> **小结：** 先在父图挂载 Checkpointer，再通过父图运行子图；状态共享由 Schema 决定，检查点隔离由运行时命名空间决定。

---

## 5. 状态查询：从父图历史找到子图快照

### 5.1 get_state 返回的不只是业务字典

```python
snapshot = graph.get_state(config)
print(snapshot.values)
print(snapshot.next)
```

本例运行完成后的结果是：

```text
{'foo': 'hi! foobar'}
()
```

返回对象是 `StateSnapshot`。读它时，先认出以下字段：

| 字段 | 本例中的含义 |
| --- | --- |
| `values` | 此快照可见的状态值 |
| `next` | 从此位置继续时计划执行的节点；结束时为空元组 |
| `config` | 定位此快照的配置，包含检查点 ID |
| `parent_config` | 前一个检查点的配置，不是“父图业务状态” |
| `tasks` | 此步骤的任务信息，可能含子图状态引用、结果或中断信息 |
| `metadata`、`created_at` | 检查点的附加信息与创建时间 |

### 5.2 从历史里找到准备进入 node_2 的位置

示例12使用：

```python
state_with_subgraph = [
    s for s in graph.get_state_history(config) if s.next == ("node_2",)
][0]
```

在本地内存后端中，历史按从新到旧的顺序返回。第一次运行得到的父图历史可以简化为：

| 从新到旧 | `values` | `next` |
| --- | --- | --- |
| 最终快照 | `{"foo": "hi! foobar"}` | `()` |
| `node_1` 完成后 | `{"foo": "hi! foo"}` | `("node_2",)` |
| 输入已合并 | `{"foo": "foo"}` | `("node_1",)` |
| 本轮输入处理前 | `{}` | `("__start__",)` |

因此，代码选中的父图快照是“父图已经加好前缀，接下来要执行子图”的位置。

`get_state_history()` 是读取历史，不会重跑历史节点。若同一线程多次运行，可能找到多个匹配项；本例的 `[0]` 取最新的匹配项。如果修改图结构后再也没有这个位置，直接取 `[0]` 会报错，应先确认搜索结果。

### 5.3 tasks 中的 state 是子图配置引用

```python
subgraph_config = state_with_subgraph.tasks[0].state
print(subgraph_config)
print(graph.get_state(subgraph_config).values)
```

本例取到的配置形如：

```python
{
    "configurable": {
        "thread_id": "1",
        "checkpoint_ns": "node_2:<任务ID>",
    }
}
```

这里的 `tasks[0].state` 是配置引用，不是 `{"foo": ..., "bar": ...}` 业务字典。把它交给 `get_state()`，才得到子图的 `StateSnapshot`。

几个定位字段可以这样区分：

- `thread_id`：选哪个逻辑会话。
- `checkpoint_ns`：选父图或哪次子图调用对应的检查点空间。
- `checkpoint_id`：在该空间中选定某个具体检查点；不指定时查询最新状态。

示例中的子图配置没有 `checkpoint_id`。整个流程此时已经跑完，所以最后查到的是该子图调用的最新状态：

```text
{'foo': 'hi! foobar', 'bar': 'bar'}
```

**父图快照位于子图执行前，不表示随后查到的子图状态也停留在执行前。** 一个保存的是父图当时的位置，一个通过引用查询子图命名空间的最新记录。历史任务中还可能附带后来完成的结果，不必把整个打印对象当作一张冻结的递归字典。

想看子图内部每一步，可以继续查询其历史：

```python
for snapshot in graph.get_state_history(subgraph_config):
    print(snapshot.values, snapshot.next)
```

本例从新到旧依次为：

```text
{'foo': 'hi! foobar', 'bar': 'bar'} ()
{'foo': 'hi! foo', 'bar': 'bar'} ('subgraph_node_2',)
{'foo': 'hi! foo'} ('subgraph_node_1',)
{} ('__start__',)
```

每个历史条目自己的 `config` 可用于定位那个具体检查点。任务 ID、检查点 ID 和时间戳每次运行不同，阅读时重点看 `values`、`next` 和命名空间。

> **小结：** 先用父图历史找到子图任务，再用任务中的配置定位子图。区分“历史快照的值”和“引用指向的最新状态”。

---

## 6. 示例13：用 Store 共享跨线程记忆

### 6.1 thread_id 和 user_id 为什么同时出现

原文件的两轮配置分别是：

```python
# 第一轮：用户1，在会话1中写入记忆。
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# 第二轮：仍是用户1，但改为会话3。
config = {"configurable": {"thread_id": "3", "user_id": "1"}}
```

Checkpointer 根据 `thread_id` 定位会话；节点自己读取 `user_id` 来构造 Store 命名空间：

```python
user_id = config["configurable"]["user_id"]
namespace = ("memories", user_id)
```

`user_id` 是示例自定义的业务配置，LangGraph 不会因为看到这个名字就自动把所有数据按用户隔离。`("memories", "1")` 中的两个元素及其顺序也是应用约定，读写时必须一致。

### 6.2 编译时同时挂载两种存储

```python
in_memory_store = InMemoryStore(
    index={
        "embed": getEmbedding(),
        "dims": 1024,
    }
)

graph = builder.compile(
    checkpointer=MemorySaver(),
    store=in_memory_store,
)
```

`embed` 提供文本向量化能力，`dims` 描述所用向量的维度。原代码的 `1024` 是该示例对嵌入模型输出的约定，不是任何模型都适用的固定值，也不是文本长度或最多保存的记忆条数。更换模型时必须核对真实输出维度。

本目录沿用这种节点签名：

```python
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    ...
```

`state` 接收图状态，`config` 接收调用配置，`store` 由运行时传入编译时挂载的存储对象。`*` 是 Python 的仅限关键字参数标记；`BaseStore` 描述存储接口，实际对象仍然是 `InMemoryStore`。

这是本地版本已验证可运行的源码写法。官方的新示例也使用 `Runtime` 读取运行时 Store；阅读其他版本时应成套理解接口，不必为了学习本目录额外迁移代码。

### 6.3 一条记忆由 namespace、key、value 组成

```python
store.put(namespace, str(uuid.uuid4()), {"data": last_message.content})
```

三个参数的职责是：

| 参数 | 示例值 | 作用 |
| --- | --- | --- |
| `namespace` | `("memories", "1")` | 组织用户1的记忆记录 |
| `key` | 随机 UUID 字符串 | 标识该命名空间中的一条记录 |
| `value` | `{"data": "你好！remember：我的名字是张三"}` | 实际保存的业务内容 |

`data` 也是示例自定义的字段，不是框架强制的固定名字。此处每次生成新的 UUID，因此重复说同一件事仍可能新增记录，不会自动更新已有姓名。若希望覆盖一条固定资料，需要由应用设计稳定的 key 和更新规则。

### 6.4 先检索，再写入，最后调用模型

按源码顺序，节点内部执行：

```text
读取 user_id，构造 namespace
    ↓
用最新消息检索已有记忆
    ↓
拼接 info，构造 system_msg
    ↓
若最新消息包含 remember，写入 Store
    ↓
system_msg + 当前线程的 messages → model.invoke()
    ↓
返回 AIMessage，合并进当前线程 State
```

检索与提示构建代码是：

```python
memories = store.search(namespace, query=str(state["messages"][-1].content))
info = "\n".join([d.value["data"] for d in memories])
system_msg = f"你是一个与用户交流的好助手。用户信息：{info}"
```

有索引且提供 `query` 时，Store 可以按向量相似度检索。返回的是带 `value` 等信息的检索结果对象，不是直接返回字符串。`info` 由业务代码从每条记录的 `data` 中提取并拼接。

之后才判断写入条件：

```python
last_message = state["messages"][-1]
if "remember" in last_message.content.lower():
    store.put(namespace, str(uuid.uuid4()), {"data": last_message.content})
```

首次运行时 Store 为空，所以 `info` 为空。即使这一轮随后保存了姓名，已经构造好的 `system_msg` 也不会自动重新检索；但模型仍能从当前用户消息中看到姓名。

第二轮更换线程后，节点检索到第一轮写入的数据，再把它明确加入模型输入：

```python
response = model.invoke(
    [{"role": "system", "content": system_msg}] + state["messages"]
)
```

**模型使用 Store 记忆的连接点就在这里。** 只把 Store 挂到图上，没有检索和提示拼接，模型不会自动看到存储内容。

### 6.5 用两轮实际数据区分“历史”与“记忆”

离线检查原脚本，结果如下。AI 回答使用固定替身，表中只展示可以确定的数据：

| 对比项 | 第一轮：线程1 | 第二轮：线程3 |
| --- | --- | --- |
| `user_id` | `"1"` | `"1"` |
| 用户输入 | `你好！remember：我的名字是张三` | `我叫什么名字` |
| 模型调用前的 State 消息数 | 1 | 1 |
| 系统提示中的用户信息 | 空 | 第一轮保存的完整文本 |
| 本轮结束后的 State 消息数 | 2 | 2 |

第二轮没有恢复第一轮的完整对话。它只有自己的新问题，却通过系统提示收到共享记忆。源码临时构造的系统消息也没有被节点返回，所以不会作为独立消息写进 `MessagesState`。

这次离线验证确认了“记录被保存、被检索、被放进模型输入”，没有验证真实嵌入模型的召回质量或真实 LLM 的回答准确率。只有一条记忆的演示，也不足以说明大量记录时的检索效果。

### 6.6 无 query 的 search 用来观察存储

原脚本中间还调用了：

```python
for memory in in_memory_store.search(("memories", "1")):
    print(memory.value)
```

没有 `query` 时，这里用于查看该命名空间前缀下的记录，而不是按“我叫什么名字”做语义排序。`search()` 有数量限制与分页参数，数据多时不能把一次无参数搜索当成“完整导出全部记忆”。

> **小结：** 跨线程记忆来自共享 Store、稳定的用户命名空间和显式的读取逻辑；线程消息历史仍由 Checkpointer 分开保存。

---

## 7. 容易混淆的边界与排查顺序

### 7.1 保存状态，不代表所有字段都会累加

示例11和13的 `messages` 使用消息 reducer，示例12的 `foo: str` 没有追加 reducer。对示例12重复提交 `{"foo": "foo"}` 时，新输入会覆盖该字段，最终仍然得到 `"hi! foobar"`，不会自动变成 `"hi! hi! foobarbar"`。

Checkpointer 负责保存与恢复，reducer 负责合并，这两个职责不能互相替代。

### 7.2 namespace 与 checkpoint_ns 不是同一个东西

| 名字 | 来源 | 用途 |
| --- | --- | --- |
| `checkpoint_ns` | 图运行时使用的检查点配置 | 区分父图与子图等检查点空间 |
| `namespace` | 示例13中的 Python 局部变量 | 组织 Store 的业务记录 |
| 流式事件的 `namespace` | `subgraphs=True` 返回的事件路径 | 标识更新发生在哪一层图 |

它们都在表达范围，但不能把 Store 的 `("memories", user_id)` 直接当成 `get_state()` 所需的配置。

### 7.3 换 user_id 不会自动更换线程历史

示例13的 Checkpointer 不是按 `(user_id, thread_id)` 组合自动隔离。若两个用户误用同一个 `thread_id`，即使节点查询了不同的 Store 命名空间，也可能加载到同一份消息历史。

实际应用应为会话分配合适的标识，并确认当前用户有权访问该会话和 Store 记录。ID 和命名空间本身不承担身份认证。

### 7.4 remember 只是字符串条件

当前写入规则不做自然语言意图识别，也不提取结构化姓名：

- “请记住我的名字是张三”没有英文 `remember`，不会触发这条分支。
- `REMEMBER` 会被 `.lower()` 转成小写后匹配。
- “do not remember this”也包含该子串，仍会写入。
- 存储的是整条用户文本，假设消息内容是字符串；多模态内容需要额外处理。

因此它是用来展示写入时机的 demo 条件，不是完整的长期记忆管理策略。

### 7.5 Store 写入与模型成功不是同一件事

示例13先 `store.put()`，再 `model.invoke()`。如果模型调用失败，不能假设前面的 Store 写入会自动撤销；若再次执行写入段，新的 UUID 还可能形成重复记录。

将来扩展为真实业务时，应按需求设计稳定键、去重或幂等写入。记忆更新、删除、冲突处理也需要业务代码负责。

### 7.6 按现象定位问题

| 现象 | 优先检查 |
| --- | --- |
| 同一会话记不住上一轮 | 是否复用 Checkpointer、是否使用相同 `thread_id`、消息如何合并 |
| 新会话取不到用户记忆 | Store 是否还是原对象、`user_id` 和命名空间是否一致、是否真的执行过 `put()` |
| Store 有数据，回答却没使用 | 检索结果是否包含它，`info` 是否进入实际模型输入 |
| 查不到子图内部状态 | 父图是否配置 Checkpointer、是否通过父图运行、是否拿到正确子图配置 |
| 向量检索失败或效果不对 | 嵌入服务配置、模型输出维度、查询文本与写入内容 |
| 还没看到示例输出就报错 | 顶层模型初始化、绘图服务及图片路径是否先失败 |
| 重启程序后全部忘记 | 当前用的是内存后端，是否需要真正落盘的实现 |

> **小结：** 顺着“状态合并 → 定位配置 → 存储生命周期 → 检索结果 → 模型输入”排查，比只观察最后一句回答更可靠。

---

## 8. 运行与练习：离线验证三个示例

### 8.1 运行原文件之前先确认依赖与副作用

从仓库根目录查看版本：

```bash
.venv/bin/python -c 'from importlib.metadata import version; print(version("langgraph")); print(version("langgraph-checkpoint"))'
```

当前 `requirements.txt` 未直接列出 `langgraph` 的固定版本，上面的本地版本说明不等于依赖文件已经锁定它。如果新环境无法导入，需要先补齐对应依赖。

| 文件 | 真实运行时的依赖与副作用 |
| --- | --- |
| 示例11 | `getModel()` 初始化聊天模型，两轮调用会访问模型服务 |
| 示例12 | 无模型调用；运行时会绘制两张 Mermaid PNG |
| 示例13 | 初始化聊天模型和嵌入模型，检索与写入可能调用嵌入服务；还会绘图 |

模型配置来自 [model.py](../../../langchaindemo/model.py)：聊天模型使用 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`；嵌入模型使用 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`。当前 `getEmbedding()` 还优先读取 `OPENAI_EMBEDDING_BASE_URL`，若配置了它，会覆盖 `EMBEDDING_BASE_URL`。

配置好服务后，示例11可从仓库根目录执行：

```bash
.venv/bin/python -m src.agent.lang_graph.persistence.示例11_如何为图添加线程级持久性
```

示例12和13中的图片路径写成 `../../../../assets/...`，它相对于**进程工作目录**解析，不是相对于 Python 文件。保留源码写法时，可以在一个子 shell 中切到示例目录，并设置项目导入路径：

```bash
(
  cd src/agent/lang_graph/persistence
  PYTHONPATH=../../../.. ../../../../.venv/bin/python 示例12_如何为子图添加现成持久特性.py
)
```

执行示例13时将最后的文件名换成 `示例13_如何为如图添加跨线程级持久性.py`，并先配置模型和嵌入服务。当前默认 Mermaid PNG 绘图可能访问外部服务；只验证状态逻辑时，使用下面的离线方式即可。

### 8.2 不修改源码，检查三个示例的数据流

以下命令从仓库根目录运行：它加载三个原始文件，临时替换模型工厂和绘图方法，再断言状态、检索内容与模型入参。替身嵌入统一返回 1024 维常量向量，**只检查存取链路，不衡量语义相似度**。

```bash
.venv/bin/python - <<'PY'
import contextlib
import io
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables.graph import Graph


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class RecordingModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content="离线模拟回答")


model_module = ModuleType("src.langchaindemo.model")
model_module.getModel = RecordingModel
model_module.getEmbedding = LocalEmbeddings
base = Path("src/agent/lang_graph/persistence")


def load_example(filename):
    # 原文件有顶层执行，收起演示打印，保留其真实图与存储对象。
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(base / filename))


with (
    patch.dict(sys.modules, {"src.langchaindemo.model": model_module}),
    patch.object(Graph, "draw_mermaid_png", return_value=b""),
):
    first = load_example("示例11_如何为图添加线程级持久性.py")
    graph = first["graph"]
    config = first["config"]
    assert [len(call) for call in first["model"].calls] == [1, 3]
    assert len(graph.get_state(config).values["messages"]) == 4
    other = graph.invoke(
        {"messages": [("user", "我的名字是什么")]},
        {"configurable": {"thread_id": "new-thread"}},
    )
    assert len(other["messages"]) == 2
    assert len(first["model"].calls[-1]) == 1
    print("示例11：同线程累计，新线程隔离，通过")

    second = load_example("示例12_如何为子图添加现成持久特性.py")
    graph = second["graph"]
    config = second["config"]
    assert graph.get_state(config).values == {"foo": "hi! foobar"}
    assert graph.get_state(config).next == ()
    history = list(graph.get_state_history(config))
    before_subgraph = next(s for s in history if s.next == ("node_2",))
    assert before_subgraph.values == {"foo": "hi! foo"}
    subconfig = before_subgraph.tasks[0].state
    assert subconfig["configurable"]["checkpoint_ns"].startswith("node_2:")
    assert graph.get_state(subconfig).values == {
        "foo": "hi! foobar", "bar": "bar"
    }
    subhistory = list(graph.get_state_history(subconfig))
    assert any(s.next == ("subgraph_node_2",) for s in subhistory)
    assert graph.invoke({"foo": "foo"}, config) == {"foo": "hi! foobar"}
    print("示例12：父子图快照、子图历史和覆盖规则，通过")

    third = load_example("示例13_如何为如图添加跨线程级持久性.py")
    graph = third["graph"]
    store = third["in_memory_store"]
    calls = third["model"].calls
    remembered = "你好！remember：我的名字是张三"
    assert [item.value for item in store.search(("memories", "1"))] == [
        {"data": remembered}
    ]
    assert calls[0][0]["content"] == "你是一个与用户交流的好助手。用户信息："
    assert remembered in calls[1][0]["content"]
    assert len(calls[1]) == 2  # 系统提示 + 新线程的一条用户消息。
    assert calls[1][1].content == "我叫什么名字"
    for thread_id in ("1", "3"):
        config = {"configurable": {"thread_id": thread_id}}
        assert len(graph.get_state(config).values["messages"]) == 2
    assert store.search(("memories", "2")) == []
    with contextlib.redirect_stdout(io.StringIO()):
        graph.invoke(
            {"messages": [("user", "我叫什么名字")]},
            {"configurable": {"thread_id": "user2-thread", "user_id": "2"}},
        )
    assert third["model"].calls[-1][0]["content"] == (
        "你是一个与用户交流的好助手。用户信息："
    )
    print("示例13：跨线程读取、模型提示注入和用户命名空间隔离，通过")

print("全部断言通过；未调用真实模型、嵌入或绘图服务")
PY
```

这段验证不需要真实 API 凭据，不修改原始示例，也不创建图片。`runpy.run_path()` 会运行顶层代码，因此替换必须在加载示例之前生效。

### 8.3 三个小练习

1. 在第 2.3 节中把第二轮的 `thread_id` 改为 `"2"`，先推导每次节点接收的消息数量，再运行核对。
2. 使用示例12的 `subgraph_config` 打印内部历史，找到只有 `foo`、还没有 `bar` 的快照，再用该条目的 `config` 精确读取它。
3. 在示例13的离线环境里，分别测试“新线程、同用户”和“新线程、新用户”；观察模型收到的系统提示，而不只观察替身回答。

> **小结：** 用确定的状态、消息数量和模型入参验证持久化行为；用真实模型与真实向量服务另行验证回答效果和检索质量。

---

## 9. 简单 Agent：完整保留模型与工具消息

### 9.1 从单次回答扩展成工具循环

[简单Agent聊天消息管理.py](简单Agent聊天消息管理.py) 在示例11的基础上增加了 `ToolNode` 和条件边：

```text
START → agent → should_continue
          ↑          ├─ 有 tool_calls → action ─┐
          └─────────────────────────────────────┘
                     └─ 无 tool_calls → END
```

`agent` 调用模型，`action` 执行工具。`should_continue` 是条件边的路由函数，不是独立注册的节点。

```python
tools = [web_search]
tool_node = ToolNode(tools)
bound_model = model.bind_tools(tools)


def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    return {"messages": response}


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "action"
```

这里两处使用 `tools`，作用不同：`bind_tools()` 告诉模型有哪些工具可请求，`ToolNode` 负责真正执行对应函数。只有模型产生工具调用消息，路由才会进入 `action`。

当前的 `web_search()` 不访问网络，不根据问题检索资料，始终返回 `"上海阳光明媚"`。它是工具流程的占位实现，返回值不能作为真实天气信息使用。

### 9.2 工具调用会在 State 中留下什么

若模型请求一次 `web_search`，正常的消息顺序为：

```text
HumanMessage：上海天气怎么样
AIMessage：tool_calls=[{name: web_search, id: 某个调用ID, args: ...}]
ToolMessage：上海阳光明媚，tool_call_id=同一个调用ID
AIMessage：根据工具结果生成的最终回答
```

第一条 AI 消息中的 `tool_calls` 是请求，`ToolMessage` 是执行结果；`ToolMessage.tool_call_id` 关联的是请求中的工具调用 ID，不是该 `AIMessage` 自身的消息 ID。

随后 `action → agent` 再次调用模型时，会把这些消息一起传回模型，让它看到“提出了什么请求、得到了什么结果”。`MessagesState` 合并的是整段交互，所以 Checkpointer 也能保存工具过程。

这与 [LangGraph 导读](../../08_LangGraph导读.md) 的外层包装示例不同：那里只把内层 Agent 最后一条回答写回外层 State；当前文件把模型和工具显式编排进同一张图，工具消息本身也属于本图状态。

### 9.3 原文件默认演示了什么

原脚本只有“你好，我是张三”和“我叫什么名字”两轮输入，并没有强制走工具分支。是否调用工具由模型输出决定，不能看到 `ToolNode` 就认定演示一定执行了搜索。

在不调用工具的确定性离线测试中：

| 时机 | 模型收到的消息数 | 本轮结束后的消息数 |
| --- | --- | --- |
| 第一轮自我介绍 | 1 | 2 |
| 第二轮询问姓名 | 3 | 4 |

另用替身模型明确返回工具调用后，验证得到的角色序列是 `human → ai → tool → ai`。此处只证明消息和节点的衔接正确，不证明真实模型一定会选择该工具。

> **小结：** 完整 Agent 历史既包含用户与最终回答，也可能包含工具请求和工具结果。后面的裁剪、删除都需要考虑这种消息结构。

---

## 10. 函数修剪：只改变本次模型输入

### 10.1 messages[-1:] 裁掉了哪一份数据

[通过函数修剪消息.py](通过函数修剪消息.py) 保留相同的 Agent 图，但在模型调用前增加了一个函数：

```python
def filter_messages(messages: list):
    return messages[-1:]


def call_model(state: MessagesState):
    messages = filter_messages(state["messages"])
    response = bound_model.invoke(messages)
    return {"messages": response}
```

`messages[-1:]` 返回只含最后一个元素的新列表；与 `messages[-1]` 返回单个消息对象不同。原代码中的两行 `print()` 只是帮助观察裁剪前后的数据。

这里没有返回删除指令，也没有把旧消息从 State 中移除。节点仍然只返回新的 AI 消息，由 reducer 追加到完整状态中：

```text
持久化 State：[Human1, AI1, Human2]
                    │ filter_messages()
                    ▼
本次模型输入：[Human2]
                    │ 得到 AI2，返回状态更新
                    ▼
持久化 State：[Human1, AI1, Human2, AI2]
```

所以这个示例降低的是本次模型看到的历史长度，**不会缩短 Checkpointer 中的当前消息列表，也不会减少已保存历史检查点的数量**。

### 10.2 为什么状态里有姓名，模型仍可能回答不出来

第二轮“我叫什么名字”到达时，State 仍有第一轮自我介绍，但 `filter_messages()` 只把新问题交给模型。

| 对比项 | 简单 Agent | 函数修剪示例 |
| --- | --- | --- |
| 第二轮 `call_model` 收到的 State | 3 条消息 | 3 条消息 |
| 第二轮实际传给模型 | 3 条消息 | 1 条消息 |
| 第二轮结束后的 State | 4 条消息 | 4 条消息 |
| 模型输入中是否有第一轮姓名 | 有 | 没有 |

因此“持久化里保存了”不等于“这次模型看到了”。真实模型是否回答出姓名不能只靠这一点预测，但程序已经没有把那条自我介绍作为本轮上下文发送。

此文件的 `thread_id="2"` 也不是遗忘的原因。每个示例都创建了自己的 `MemorySaver`，不能把不同脚本中的线程数字当成同一个存储里的连续实验。

### 10.3 工具结果不能只留最后一条

这是当前修剪函数最需要注意的边界。若第一步请求工具，工具执行后状态是：

```text
[HumanMessage, AIMessage(tool_calls=...), ToolMessage(...)]
```

回到 `agent` 再次执行 `messages[-1:]`，模型只会收到：

```text
[ToolMessage(...)]
```

这样就丢掉了对应的 AI 工具调用请求。对要求工具调用配对的模型接口，这种输入可能被拒绝；即使某个替身模型接受了，也不能证明真实接口会接受。

更合理的裁剪策略需要保留有效的消息片段，例如近期完整对话，以及所有保留的工具结果对应的请求。固定保留“最后 N 条”并不保证配对正确；并行调用多个工具时尤其不能只靠一对相邻消息判断。

当前函数也没有计算 Token，不能把它解释为“按模型上下文预算自动修剪”。一条很长的消息仍可能超出预算，系统提示和工具定义也可能占用上下文。

> **小结：** 函数裁剪改变模型输入视图，不改变持久化状态；裁剪边界还必须保持消息协议完整。

---

## 11. 删除节点：通过 RemoveMessage 更新状态

### 11.1 删除发生在回答之后

[删除消息节点.py](删除消息节点.py) 将“没有工具请求”时的出口改成删除节点：

```text
START → agent → should_continue
          ↑          ├─ 有 tool_calls → action ─┐
          └─────────────────────────────────────┘
                     └─ 无 tool_calls → delete_messages → END
```

核心函数为：

```python
def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

`RemoveMessage` 不是发给模型的普通文本消息。`MessagesState` 使用的 `add_messages` 会识别这些指令，按消息 ID 从当前列表中移除对应项。图合并消息时会给缺少 ID 的消息补充 ID，因此节点能引用当前状态里的 `m.id`。

消息数不超过两条时，函数隐式返回 `None`；在当前图中表示不提交状态更新，随后沿固定边结束。

### 11.2 删除前能看到，删除后不再保留

先只看每轮产生一条普通回答、没有工具调用的场景：

| 时机 | 状态或行为 |
| --- | --- |
| 第一轮回答完成 | `[Human1, AI1]`，长度为 2，不删除 |
| 第二轮模型调用前 | `[Human1, AI1, Human2]`，模型仍能看到旧姓名 |
| 第二轮回答完成 | `[Human1, AI1, Human2, AI2]`，长度为 4 |
| 删除节点完成 | 移除 `Human1`、`AI1`，最终为 `[Human2, AI2]` |

因此，这个节点不会减少第二轮已经发生的模型调用成本，它是在回答之后调整后续调用可读取的 State。若最新 AI 回答本身复述了姓名，姓名仍可能留在最近消息中；删除旧自我介绍不等于保证删除所有提及姓名的文本。

原脚本使用 `stream_mode="values"`，所以观察过程中可能先看到四条消息的状态，再看到删除后的两条。应使用 `app.get_state(config).values` 核对最终保存的当前状态，而不是把某个中间事件当作最终结果。

### 11.3 “删除前两条”不等于“只保留最后两条”

当前表达式是 `messages[:2]`：当消息数大于 2 时，每次只删最早两条。

| 删除前的消息数 | 当前函数删除数 | 删除后的消息数 |
| --- | --- | --- |
| 2 | 0 | 2 |
| 4 | 2 | 2 |
| 8 | 2 | 6 |

后面的摘要示例使用的是 `messages[:-2]`，意思是删除最后两条之前的所有消息，两种切片不能混淆。

工具交互也会增加消息数。若机械删除最早两条，可能删掉用户输入和 AI 工具请求，却留下孤立的 `ToolMessage`。实际保留窗口应按有效对话和工具调用边界组织，不能只看列表长度。

### 11.4 源码绑定了工具，但调用的仍是原始 model

原文件同时写了：

```python
bound_model = model.bind_tools(tools)


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}
```

此处 `bound_model` 没有被节点使用。不能因为前面执行过 `bind_tools()`，就假设后面的原始 `model.invoke()` 已经带上该工具定义。离线检查也确认两轮调用都进入原始 `model`，绑定后的对象调用次数为 0。

因此本文件当前的重点是普通聊天后的消息删除。若希望像第 9 节一样展示工具循环，需要在节点中实际使用 `bound_model.invoke()`，并同时处理工具消息的保留边界；本文说明这一处源码差异，没有替用户改动示例实现。

### 11.5 删除当前消息不等于清除检查点历史

删除节点产生的是一次新的状态更新。`get_state(config)` 看到的最新消息列表变短，但之前的检查点仍可能包含被删消息。本地验证中，删除示例的最新状态只有两条，历史中仍能找到含四条消息的快照。

所以 `RemoveMessage` 适合管理后续运行使用的状态；如果目标是彻底清理存储数据，还要考虑历史检查点和后端的数据清理机制。

> **小结：** 删除节点通过 reducer 真正修改当前消息状态，但执行时机、切片范围、工具消息配对和历史检查点都要分别理解。

---

## 12. 会话摘要：把旧消息压缩到 summary

### 12.1 新增一个状态字段保存摘要

[如何添加会话历史摘要.py](如何添加会话历史摘要.py) 没有接入工具，而是在聊天节点后增加摘要分支：

```text
START → conversation → should_continue
                           ├─ len(messages) ≤ 6 → END
                           └─ len(messages) > 6 → summarize_conversation → END
```

状态继承自 `MessagesState`：

```python
class State(MessagesState):
    summary: str
```

`messages` 继续使用消息 reducer；`summary` 没有声明追加 reducer，新摘要会替换旧摘要。`TypedDict` 声明不自动生成初始值，所以代码使用 `state.get("summary", "")` 读取尚不存在的字段。

两部分都由当前线程的 Checkpointer 保存。这个文件没有 Store：`summary` 是线程内的压缩历史，不会自动变成跨线程用户记忆。

### 12.2 先用旧摘要回答，再决定是否生成新摘要

聊天节点如下：

```python
def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earliear:{summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}
```

这里沿用原代码中 `earliear` 的拼写，它只是提示文本，不是 API 参数。真正影响数据流的是：存在摘要时，临时在模型输入前加一条系统消息。

这条系统消息没有作为节点更新返回，所以不会长期追加到 `state["messages"]`；下次调用时根据最新 `summary` 重新构造。

`should_continue()` 在聊天回答已经合并进 State 后检查 `len(messages) > 6`。这是**消息条数阈值**，不是 Token 预算，也不是用户发言超过六轮。当前没有工具，每轮恰好一问一答时，第三轮后为 6 条，第四轮后为 8 条，才首次触发摘要。

### 12.3 首次摘要与后续摘要如何构造

摘要节点读取当前摘要，然后在现有消息末尾追加一条临时请求：

```python
summary = state.get("summary", "")
if summary:
    summary_messages = (
        f"这是迄今为止的对话摘要：{summary}\n\n"
        "通过考虑上述新消息来扩展摘要："
    )
else:
    summary_messages = "创建上述对话的摘要"

messages = state["messages"] + [HumanMessage(content=summary_messages)]
response = model.invoke(messages)
delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
return {"summary": response.content, "messages": delete_messages}
```

首次总结时，模型看到现有全部对话加摘要请求；以后总结时，请求里同时带上旧摘要，要求结合当前保留的新消息更新它。

该节点提交两种更新：

1. 将模型生成的文本写入 `summary`，覆盖旧值。
2. 删除 `messages` 中除最后两条之外的旧消息。

摘要请求本身，以及模型生成的摘要回答，都没有作为普通对话消息追加到 State。摘要回答的内容进入了独立的 `summary` 字段。

注意：模型总结的范围是当前的**全部**消息，最后两条也参与总结，同时这两条仍然保留在 State。因此摘要内容与最近消息存在重叠，这是当前实现的实际行为，不能描述成“只总结将被删除的部分”。

### 12.4 沿着原脚本的六轮输入走一遍

在每轮返回一条普通 AI 消息的条件下，原脚本的确定性状态变化为：

| 轮次 | 用户输入 | 回答后、检查阈值时 | 本轮结束后的消息数 | 摘要行为 |
| --- | --- | --- | --- | --- |
| 1 | 你好，我是张三 | 2 | 2 | 不生成 |
| 2 | 我叫什么名字 | 4 | 4 | 不生成 |
| 3 | 我喜欢曼联！ | 6 | 6 | 恰好 6，不生成 |
| 4 | 我喜欢他们总是赢球 | 8 | 2 | 首次生成，删除最早 6 条 |
| 5 | 我叫什么名字 | 4 | 4 | 使用旧摘要回答，不重新生成 |
| 6 | 你觉得我喜欢哪支英超球队？ | 6 | 6 | 使用旧摘要回答，仍未超过阈值 |

第四轮结束时的结构可概括为：

```text
summary：前四轮对话的压缩信息
messages：[Human4, AI4]
```

第五轮回答前，模型收到的是：

```text
[SystemMessage(summary), Human4, AI4, Human5]
```

原来的自我介绍不在近期消息中，模型需要从摘要找回相关事实。摘要是否真的保留姓名和球队，要通过真实模型结果检查；离线替身只保证我们能验证它被正确传入。

六轮总共调用模型 **7 次**：6 次普通回答，第四轮结束后额外调用 1 次摘要。摘要能减少后续的原始历史长度，但本身也需要一次模型调用；触发它的第四轮回答已经用过较长的历史，不能将摘要理解为在每次模型请求前强制限制上下文。

### 12.5 updates 流中的删除指令不是最终消息列表

本文件使用 `stream_mode="updates"`，并按节点输出逐条打印。第四轮通常产生两类更新：

```text
conversation:
  messages = [本轮 AI 回答]

summarize_conversation:
  summary = 新摘要
  messages = [RemoveMessage(...), ...]
```

`print_update()` 打印到 `RemoveMessage`，说明看到了节点提交的删除更新，并不表示这些删除标记留在最终聊天记录中。更新经过 reducer 后，最终 `messages` 只保留最近两条，可通过 `app.get_state(config).values` 确认。

如果改为 `values`，观察到的则是合并后的完整状态，不应再使用按节点名拆解 `updates` 的读取方式。

### 12.6 末尾的 update_state 实际只删除一条

六轮执行完，原脚本继续执行：

```python
messages = app.get_state(config).values["messages"]
app.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})
```

此时列表是 `[Human4, AI4, Human5, AI5, Human6, AI6]`，删除首条后变成：

```text
[AI4, Human5, AI5, Human6, AI6]
```

因此原脚本最终保存 **5 条消息**，`summary` 保持不变。这次 `update_state()` 将更新交给 reducer，并生成新的检查点，本身不调用模型，也不自动重新运行摘要节点。

源码附近提到 `REMOVE_ALL_MESSAGES`，但实际传入的是 `messages[0].id`，两者不同。若要在已有 `app`、`config` 的上下文中清空当前消息列表，扩展写法是：

```python
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

app.update_state(config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})
```

这也只清空当前 `messages`，不会顺便清空 `summary`、Store 或旧检查点。若普通 `RemoveMessage` 指向不存在的消息 ID，当前 reducer 会报错；不要将它当作按文本内容模糊删除的 API。

单独删除一条消息还可能让上下文从 AI 回答开头，工具场景中则可能破坏调用配对。对消息格式有要求的模型接口，需要在下一次调用前检查保留序列。

### 12.7 摘要是有损压缩，不是原文备份

当前实现有几个适合继续练习的边界：

- 阈值只计算消息条数，长消息和不断增长的 `summary` 仍会增加上下文开销。
- 摘要内容由模型生成，可能漏掉姓名、偏好、时间等细节；可以在提示中明确需要保留的字段。
- 后续摘要依赖旧摘要和近期消息，早期遗漏的信息不会仅靠重复总结自动恢复。
- 原始内容可能仍存在旧检查点里，压缩当前 State 不等于清理历史存储。

> **小结：** 摘要节点同时更新 `summary` 和近期消息窗口；聊天节点负责重新把摘要放进模型输入。两步缺一不可。

---

## 13. 新增示例的离线验证与练习

### 13.1 四个文件的运行条件

四个文件都在顶层初始化模型、绘制 Mermaid PNG 并执行演示，导入也会触发这些动作。三个 Agent 文件中的 `web_search` 虽然只是本地函数，聊天模型本身仍是外部依赖；运行原文件前要配置第 8.1 节中的模型参数。

所有图片路径继续使用 `../../../../assets/...`，工作目录的要求与第 8.1 节一致。另外，修剪示例的输出文件名没有 `.png` 后缀，虽然调用的仍是 `draw_mermaid_png()`，查找输出时要按源码中的文件名定位。

下面优先验证状态与消息协议，不依赖 API 凭据或绘图网络。替身会记录实际输入，遇到明确的天气问题时请求本地工具，并为摘要请求返回固定文本；固定摘要用于断言数据流，不代表模型自然语言总结能力。

### 13.2 可复制的离线验证命令

在仓库根目录运行：

```bash
.venv/bin/python - <<'PY'
import contextlib
import io
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables.graph import Graph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages


class RecordingModel:
    def __init__(self, bound=False):
        self.bound = bound
        self.calls = []
        self.summary_count = 0

    def bind_tools(self, tools):
        # 分开记录绑定前后对象，检查节点究竟调用了哪一个。
        return RecordingModel(bound=True)

    def invoke(self, messages):
        self.calls.append(list(messages))
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(content="工具处理完成")
        if (
            last.content == "创建上述对话的摘要"
            or last.content.startswith("这是迄今为止的对话摘要：")
        ):
            self.summary_count += 1
            return AIMessage(content=f"摘要{self.summary_count}：用户叫张三，喜欢曼联")
        if self.bound and last.content == "上海天气怎么样":
            return AIMessage(content="", tool_calls=[{
                "name": "web_search",
                "args": {"query": last.content},
                "id": "weather-call",
                "type": "tool_call",
            }])
        return AIMessage(content="离线模拟回答")


module = ModuleType("src.langchaindemo.model")
module.getModel = RecordingModel
base = Path("src/agent/lang_graph/persistence")


def load_example(filename):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(base / filename))


def weather_turn(example):
    with contextlib.redirect_stdout(io.StringIO()):
        return example["app"].invoke(
            {"messages": [("user", "上海天气怎么样")]},
            {"configurable": {"thread_id": "weather-test"}},
        )


with (
    patch.dict(sys.modules, {"src.langchaindemo.model": module}),
    patch.object(Graph, "draw_mermaid_png", return_value=b""),
):
    basic = load_example("简单Agent聊天消息管理.py")
    assert [len(c) for c in basic["bound_model"].calls] == [1, 3]
    result = weather_turn(basic)
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai"]
    assert result["messages"][2].tool_call_id == (
        result["messages"][1].tool_calls[0]["id"]
    )
    assert [m.type for m in basic["bound_model"].calls[-1]] == ["human", "ai", "tool"]
    print("简单 Agent：完整历史和工具消息配对，通过")

    trimmed = load_example("通过函数修剪消息.py")
    assert [len(c) for c in trimmed["bound_model"].calls] == [1, 1]
    assert len(trimmed["app"].get_state(trimmed["config"]).values["messages"]) == 4
    weather_turn(trimmed)
    assert [m.type for m in trimmed["bound_model"].calls[-1]] == ["tool"]
    print("函数修剪：状态仍有历史；已复现工具请求丢失的边界")

    deleted = load_example("删除消息节点.py")
    app, config = deleted["app"], deleted["config"]
    assert [len(c) for c in deleted["model"].calls] == [1, 3]
    assert deleted["bound_model"].calls == []
    assert len(app.get_state(config).values["messages"]) == 2
    assert any(len(s.values.get("messages", [])) == 4 for s in app.get_state_history(config))
    eight = [HumanMessage(content=str(i), id=f"m{i}") for i in range(8)]
    update = deleted["delete_messages"]({"messages": eight})
    assert len(update["messages"]) == 2
    assert len(add_messages(eight, update["messages"])) == 6
    print("删除节点：删除时机、只删两条和历史快照保留，通过")

    summarized = load_example("如何添加会话历史摘要.py")
    app, config, model = summarized["app"], summarized["config"], summarized["model"]
    assert [len(c) for c in model.calls] == [1, 3, 5, 7, 9, 4, 6]
    assert model.summary_count == 1
    assert len(summarized["values"]["messages"]) == 2  # 第四轮结束时的快照值。
    assert model.calls[5][0].type == "system"
    assert "摘要1" in model.calls[5][0].content
    current = app.get_state(config).values
    assert len(current["messages"]) == 5  # 六轮后手动删除了第一条。
    assert current["messages"][0].type == "ai"
    assert current["summary"] == "摘要1：用户叫张三，喜欢曼联"

    # 继续同一线程：5 条旧消息 + 新问题 + 新回答 = 7，再次触发摘要。
    with contextlib.redirect_stdout(io.StringIO()):
        result = app.invoke({"messages": [("user", "继续聊球队")]}, config)
    assert model.summary_count == 2
    assert len(result["messages"]) == 2
    assert "摘要1" in model.calls[-1][-1].content
    assert result["summary"] == "摘要2：用户叫张三，喜欢曼联"

    app.update_state(config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})
    assert app.get_state(config).values["messages"] == []
    assert app.get_state(config).values["summary"] == result["summary"]
    assert model.summary_count == 2
    try:
        add_messages(eight, [RemoveMessage(id="not-found")])
    except ValueError:
        pass
    else:
        raise AssertionError("删除不存在的消息 ID 应报错")
    print("会话摘要：触发时机、摘要注入、增量更新和手动删除，通过")

print("全部离线断言通过；工具裁剪问题已复现，未验证真实模型接口")
PY
```

工具测试中的替身允许孤立的 `ToolMessage`，目的是记录修剪函数确实丢掉了请求消息，而不是认定该序列合法。验证脚本临时修改的是自己加载的内存对象，没有修改四个源文件，也不清理用户正在运行的其他进程。

### 13.3 三个小练习

1. 对比简单 Agent 和修剪示例的第二轮模型入参，再对比 `get_state()`。解释为什么 State 都有四条消息，但模型实际看到的历史不同。
2. 给删除函数传入 8 条有 ID 的消息，比较 `messages[:2]` 与 `messages[:-2]` 生成的删除集合；再换成真实的工具消息序列，检查是否破坏配对。
3. 在摘要示例中比较第三轮、第四轮、第六轮及末尾 `update_state()` 之后的消息数；继续发一轮输入，验证旧摘要进入新的摘要请求。

> **小结：** 分别断言模型入参、最新状态和历史检查点，再观察摘要调用次数，才能区分“看得少”“保存得少”和“压缩后继续使用”。

---

## 14. 按需求选择保存与消息管理方式

| 需求 | 当前示例提供的方式 | 还需要注意什么 |
| --- | --- | --- |
| 同一会话接着聊 | `MessagesState + MemorySaver + thread_id` | 复用存储对象，正确合并消息 |
| 查看父图执行到哪里 | `get_state()`、`get_state_history()` | 区分最新快照和带 ID 的历史快照 |
| 查看子图内部步骤 | 父图 Checkpointer + 子图任务配置 | 子图的命名空间及调用作用域 |
| 新会话读取用户资料 | `InMemoryStore + namespace + search/put` | 主动读写，并显式加入模型输入 |
| 跨进程或重启保留数据 | 更换为可共享、可持久保存的存储后端 | Checkpointer 与 Store 两边分别配置 |
| 从暂停处继续业务 | Checkpointer 配合中断恢复接口 | 参考前面的中断笔记，不能只调用 `get_state()` |
| 保存工具调用全过程 | 模型节点与 `ToolNode` 共用 `MessagesState` | 工具请求与结果必须正确对应 |
| 减少本次模型读取的消息 | 模型调用前使用过滤函数 | 不会自动缩短 State；不能破坏消息协议 |
| 缩短后续使用的当前历史 | 返回 `RemoveMessage`，由 reducer 删除 | 删除时机、保留范围和历史检查点是不同问题 |
| 保留旧对话要点并缩短近期消息 | `summary` 字段 + 摘要节点 + 删除旧消息 | 摘要可能遗漏信息，也有额外模型调用开销 |

可以用下面的清单复习本目录：

```text
MessagesState / reducer → 定义并合并消息
Checkpointer → 保存图状态与执行位置
thread_id → 定位逻辑会话
checkpoint_ns / checkpoint_id → 定位检查点空间与具体版本
get_state / get_state_history → 观察已保存状态
Store + namespace + key → 保存应用选择的业务记忆
search → info → system_msg → 让模型实际获得记忆
内存后端 / 持久后端 → 决定数据能否跨进程重启保留
filter_messages → 选择这一次模型看到的消息
RemoveMessage → 按 ID 修改当前消息状态
summary → 保存压缩信息，并在后续模型输入中重新注入
```

> **小结：** 先判断要保存的是执行状态还是业务资料，再区分模型输入、当前消息状态和历史检查点；用合适的保留策略控制上下文，并检查需要的信息是否真的进入模型调用。

---

## 参考示例与延伸阅读

- [示例11_如何为图添加线程级持久性.py](示例11_如何为图添加线程级持久性.py)：同线程两轮对话。
- [示例12_如何为子图添加现成持久特性.py](示例12_如何为子图添加现成持久特性.py)：父子图检查点与状态历史。
- [示例13_如何为如图添加跨线程级持久性.py](示例13_如何为如图添加跨线程级持久性.py)：用户记忆的写入、检索与模型提示。
- [简单Agent聊天消息管理.py](简单Agent聊天消息管理.py)：完整模型与工具消息循环。
- [通过函数修剪消息.py](通过函数修剪消息.py)：只保留最后一条模型输入的演示及边界。
- [删除消息节点.py](删除消息节点.py)：回答后删除最早两条消息。
- [如何添加会话历史摘要.py](如何添加会话历史摘要.py)：摘要、近期消息保留与手动状态更新。
- [初识 Graph](../base/09_初识Graph：State、Node与Edge.md)：状态与 reducer 的基础。
- [子图：状态共享、转换与父图跳转](../adv/subgraphs/12_LangGraph子图：状态共享、转换与父图跳转.md)：父子图的数据接口。
- [中断与编辑状态](../adv/interr/11_LangGraph中断与编辑状态.md)：持久化如何支撑暂停与恢复。

官方 Persistence、Memory、Subgraphs 与 Store 文档分别用于核对存储职责、消息管理、运行时注入、子图持久化模式和检索接口。本文的输出与断言以仓库代码及开头列出的本地版本为准；第 2.3 节是独立示例，第 8.2、13.2 节是可从仓库根目录执行的验证命令，其余代码片段主要用于解释所在上下文。

```text
https://docs.langchain.com/oss/python/langgraph/persistence
https://docs.langchain.com/oss/python/langgraph/add-memory
https://docs.langchain.com/oss/python/langgraph/use-subgraphs
https://docs.langchain.com/oss/python/langgraph/stores
```
