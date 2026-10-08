# LangGraph 多智能体：网络、工具交接与 Command

多智能体不是“多调用几次模型”这么简单。它要解决的是：怎样把不同职责、不同工具和不同上下文的 Agent 组织成一个可观察、可恢复、可控的工作流。

本章对应当前目录中的三个示例：

| 文件 | 示例重点 |
| --- | --- |
| [`如何构建多智能体网络.py`](如何构建多智能体网络.py) | 旅行顾问与酒店顾问之间的专家交接 |
| [`使用Command进行交接.py`](使用Command进行交接.py) | 用 `Command(goto=..., update=...)` 直接跳到另一个专家 |
| [`使用工具实现交接.py`](使用工具实现交接.py) | 用工具调用作为交接信号，并封装 Agent 的工具循环 |

这些脚本会在导入时创建模型、绘制图或发起请求。它们是学习示例，运行前需要配置模型服务；做离线测试时，应把建图和执行入口提取到函数中，并放进 `if __name__ == "__main__":`。

## 1. 这章在学习路线中的位置

可以把仓库前面的内容压缩成下面几层：

```text
RAG：让模型看到外部知识
  ↓
LangChain：把 Prompt、模型、解析器和 Retriever 组合起来
  ↓
Tool calling：让模型提出结构化的外部操作请求
  ↓
Agent：让模型在工具和回答之间循环
  ↓
LangGraph：用 State、Node、Edge 描述可恢复的流程
  ↓
多智能体：把多个有边界的 Agent 连接成网络
```

多智能体示例仍然依赖前面的基础：

- `StateGraph` 定义控制流，`MessagesState` 保存对话消息；
- `@tool` 和 `bind_tools()` 负责描述工具、生成 `tool_calls`；
- `ToolMessage.tool_call_id` 把工具结果和模型请求配对；
- reducer 决定多个节点如何合并同一个 State 字段；
- Checkpointer 与 `thread_id` 决定一次会话能否暂停、恢复和隔离；
- `stream()`、图可视化和节点日志用来观察真实路径；
- RAG、检索评估和中间件则分别提供知识、质量反馈和运行时规则。

前置笔记可按以下顺序复习：[`Tool 与 Function Calling`](../../../tool/04_Tool与FunctionCall.md)、[`Agent 中间件`](../../../middleware/05_Agent中间件.md)、[`LangGraph 导读`](../../08_LangGraph导读.md)、[`初识 Graph`](../base/09_初识Graph：State、Node与Edge.md)、[`LangGraph 工具调用`](../toolUse/15_LangGraph工具调用：ToolNode、状态注入与容错.md)。持久化、子图、中断、循环和评估分别见本目录上级的对应章节。

## 2. 先建立一个多智能体心智模型

一个 Agent 至少包含四个部分：

```text
Agent = 角色提示词 + 可见工具 + 消息历史 + 输出/路由规则
```

多个 Agent 连接后，图的状态可以这样看：

```text
用户问题
   │
   ▼
┌──────────────┐   Command / handoff   ┌──────────────┐
│ travel_agent │ ─────────────────────▶ │ hotel_agent  │
│ 选目的地      │                        │ 推荐酒店      │
└──────┬───────┘                        └──────┬───────┘
       │                                       │
       └────────────── 最终消息 ◀───────────────┘
```

箭头表示控制权和消息历史的流转。它不表示另一个 Agent 是一个普通 Python 函数，也不表示两个 Agent 会自动共享所有本地变量；它们共享什么，取决于图的 State、消息 reducer 和交接协议。

多智能体设计时要先回答三个问题：

1. **谁负责决定下一位 Agent？** 当前 Agent、独立路由节点，还是一个 supervisor？
2. **交接时要传什么？** 完整消息历史、结构化任务、工具结果，还是只传摘要？
3. **什么时候结束？** 专家回答后直接结束，回到原 Agent，还是继续经过审核和格式化节点？

## 3. 四种常见组织方式

| 方式 | 控制权 | 适合场景 | 主要代价 |
| --- | --- | --- | --- |
| 专家网络 | 当前 Agent 选择下一个 Agent | 任务路径不固定，专家之间需要互相求助 | 可能来回交接，难以限制循环 |
| Supervisor | 中央路由 Agent 选择专家 | 角色较多，需要统一调度和审计 | supervisor 本身增加一次模型调用 |
| 工具式交接 | Agent 调用 `transfer_to_*` 工具 | 想复用模型的工具选择能力 | 工具只是信号，仍需实现跳转和消息协议 |
| 子图式 Agent | 每个 Agent 是一个编译后的子图 | 角色内部还有工具循环、重试和局部状态 | 需要设计父子 State 转换和 `Command.PARENT` |

“多智能体”不等于必须使用多张图。当前 `如何构建多智能体网络.py` 把两个专家直接注册为父图节点；`使用工具实现交接.py` 则展示了如何先构造一个带工具循环的 Agent，再把它作为节点组合。

如果一个角色只有一条简单指令，普通节点就够了；如果角色内部要多次调用工具、处理异常或等待人工输入，再考虑把它封装为子图。

## 4. 示例一：构建一个专家网络

### 4.1 两个专家和两个交接工具

旅行示例先定义两个“交接工具”：

```python
@tool
def transfer_to_travel_advisor():
    """向旅行顾问寻求帮助"""
    return


@tool
def transfer_to_hotel_advisor():
    """向酒店顾问寻求帮助"""
    return
```

这两个函数的函数体几乎为空，因为它们的主要作用不是执行外部操作，而是让模型发出一个结构化信号：当前问题应交给谁。模型是否调用交接工具，由工具名、docstring、系统提示词和当前消息共同决定。

这和“查询酒店数据库”的工具不同：后者应该返回真实数据；交接工具的返回值只是图控制逻辑的一部分。不要把这两类工具混在同一个权限边界里，真正有副作用的工具仍需程序侧做参数校验、授权和审计。

### 4.2 节点同时负责回答和路由

旅行专家的核心结构是：

```python
def travel_advisor(state: MessagesState):
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_hotel_advisor]).invoke(messages)

    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="hotel_advisor",
            update={"messages": [ai_msg, tool_msg]},
        )

    return {"messages": [ai_msg]}
```

执行顺序如下：

```text
START
  ↓
travel_advisor：推荐目的地，必要时请求酒店专家
  ├─ 没有 tool_calls → 返回回答，流程结束
  └─ 有 tool_calls → 写入 AIMessage 和 ToolMessage，goto hotel_advisor
                         ↓
                    hotel_advisor：读取完整历史并推荐酒店
                         ├─ 没有 tool_calls → 返回最终回答
                         └─ 有 tool_calls → goto travel_advisor
```

这里的 `AIMessage` 必须保留完整对象。它不仅可能有文本，还包含 `tool_calls`、调用名称和调用 ID。只保存 `ai_msg.content` 会让后续节点失去交接依据。

### 4.3 为什么酒店专家要检查完整历史

酒店专家的系统提示词要求它从历史中找到旅行专家已经确定的目的地，再给出酒店建议。这体现了交接的一个关键约束：

> 交接协议要明确“下一个 Agent 可以相信哪些上下文”。

当前示例直接传递完整 `messages`，所以酒店专家可以看到用户问题、旅行专家的回答、交接工具请求和工具结果。生产实现也可以传递结构化的 `destination` 字段，但应明确由谁写入、谁校验，以及字段缺失时如何降级。

### 4.4 当前网络中的循环风险

两个专家都能调用对方的交接工具，因此图在静态上允许：

```text
travel_advisor ↔ hotel_advisor
```

如果模型反复判断“还需要另一位专家”，执行可能超过递归限制。可以按前面循环笔记的方式设置：

```python
graph.invoke(
    {"messages": [("user", "...")]},
    config={"recursion_limit": 20},
)
```

`recursion_limit` 是图步骤的硬上限，超限会抛出异常；它不是业务上的“最多交接 20 次”。更稳妥的方案是增加 State 字段记录 `handoff_count`、已访问 Agent 或任务阶段，并在到达上限时返回解释清楚的降级结果。

## 5. 示例二：用 `Command` 完成交接

### 5.1 `Command` 的两个核心字段

`Command` 把状态更新和下一步路由放在同一个返回值中：

```python
return Command(
    goto="multiplication_expert",
    update={"messages": [ai_msg, tool_msg]},
)
```

| 字段 | 作用 |
| --- | --- |
| `update` | 按 State 的 reducer 写入本次新增内容 |
| `goto` | 指定下一次执行的节点 |
| `graph` | 可选，指定命令作用在当前图还是父图 |

`使用Command进行交接.py` 的节点签名写成：

```python
def addition_expert(
    state: MessagesState,
) -> Command[Literal["multiplication_expert", "__end__"]]:
    ...
```

`Literal` 告诉类型检查器和图可视化：这个节点的动态目标只有乘法专家或结束。它不能替代运行时校验，`goto` 仍必须指向已经注册的节点。

### 5.2 交接消息为什么要成对写入

模型调用交接工具后，消息历史里至少需要保留：

1. 模型发出的 `AIMessage`，其中包含 `tool_calls`；
2. 与该调用 ID 对应的 `ToolMessage`，表示交接已被程序接受。

示例手动构造的工具消息是：

```python
tool_msg = {
    "role": "tool",
    "content": "成功转移",
    "tool_call_id": tool_call_id,
}
```

`tool_call_id` 不是装饰字段。它让模型接口知道这条工具结果对应哪一次请求。若省略它、写错它，下一位 Agent 或模型可能无法继续解析对话。

### 5.3 `Command` 与条件边的取舍

下面两种图都可以表达“加法节点之后去乘法节点或结束”：

```python
# 当前节点直接决定下一步
return Command(update={"messages": [ai_msg]}, goto="multiplication_expert")
```

```python
# 节点更新状态，单独的路由函数决定下一步
return {"messages": [ai_msg]}


def route_after_addition(state: MessagesState):
    return "multiplication_expert" if needs_multiplication(state) else END


builder.add_conditional_edges(
    "addition_expert",
    route_after_addition,
    ["multiplication_expert", END],
)
```

选择标准是：

- 路由和节点业务必须一起完成，并且两者共享同一个判断结果时，`Command` 更直接；
- 路由逻辑要复用、单独测试或由多个节点共享时，条件边更清晰；
- 需要根据列表创建多份独立任务时，使用 `Send`，不要把 `Command` 当作 MapReduce 工具。

### 5.4 `graph` 参数和父图跳转

当前三个示例的 Agent 都直接注册在同一张父图上，所以只需要 `goto="another_node"`。如果 Agent 是子图，跳到父图中的兄弟节点时，需要显式表达作用层级：

```python
return Command(
    goto="hotel_advisor",
    graph=Command.PARENT,
    update={"messages": [tool_msg]},
)
```

这时要同时检查三件事：

1. `hotel_advisor` 是否注册在父图，而不是只存在于子图内部；
2. `update` 是否符合父图的 State schema；
3. 子图到父图的消息或字段是否会被 reducer 正确合并。

子图并不会自动获得独立记忆。父图挂载 Checkpointer 后，子图的状态快照会按运行时命名空间记录；跨线程业务记忆仍应使用 Store，并通过稳定的 `user_id` 显式读取。

## 6. 示例三：把工具循环封装成 Agent

### 6.1 `make_agent()` 做了什么

`使用工具实现交接.py` 中的 `make_agent(model, tools, system_prompt=None)` 返回一张已经编译好的小图。它包含两个节点：

```text
START → call_model ── 有 tool_calls ──→ call_tools
             ↑                             │
             └──────── 工具结果 ────────────┘
             └── 无 tool_calls → END
```

`call_model`：

1. 从 `MessagesState` 取出完整消息；
2. 在需要时加入系统提示词；
3. 调用绑定了工具的模型；
4. 有工具请求时返回 `Command(goto="call_tools", ...)`；
5. 没有工具请求时返回普通消息更新。

`call_tools`：

1. 读取最后一条 `AIMessage.tool_calls`；
2. 按名称从 `tools_by_name` 找到工具；
3. 解析 `args`，必要时把 JSON 字符串转成字典；
4. 如果工具 schema 声明需要 `state`，把当前 State 注入参数；
5. 执行工具，并把 `ToolMessage` 或 `Command` 写回图。

这里同时出现了两种 `Command`：

- 模型节点的 `Command` 控制 Agent 内部去 `call_tools`；
- 某个工具返回的 `Command` 可以让工具执行后把控制权交给另一个 Agent。

前者是固定的内部工具循环，后者是跨 Agent 的交接。阅读代码时要先区分这两个层级。

### 6.2 工具参数解析不是可有可无的补丁

示例对 `args` 做了兼容处理：有些模型返回字典，有些兼容接口可能把参数序列化成 JSON 字符串。生产代码中不应直接使用宽泛的 `except:` 静默吞错；应记录工具名、校验失败原因和调用 ID，同时隐藏敏感参数，并把错误转成模型能理解的结构化 `ToolMessage`。

工具 schema 也应在边界上校验。模型提出的参数不等于已授权的参数，尤其是网址、文件路径、SQL、用户标识和写操作。可以结合中间件做统一日志、脱敏、拦截和重试，但具有副作用的工具仍需要明确的授权策略。

### 6.3 当前文件实际运行的是单 Agent 示例

文件末尾默认执行：

```python
agent = make_agent(model, [add, multiply])
for chunk in agent.stream({"messages": [("user", "(3 + 5) * 12")]}):
    pretty_print_messages(chunk)
```

这段代码验证的是一个 Agent 使用加法和乘法工具的循环。后面的多 Agent 版本被注释掉了，只有取消注释后才会创建 `addition_expert` 和 `multiplication_expert` 两个 Agent，并通过交接工具组合成父图。

因此，看到这个文件时不要把“注释中的多 Agent 代码”当作当前默认行为。先运行单 Agent，确认 `AIMessage → ToolMessage → AIMessage` 协议，再逐步打开交接部分。

### 6.4 工具交接和普通业务工具的差异

```text
普通工具：模型请求 → 程序执行函数 → 返回业务结果
交接工具：模型请求 → 程序确认交接 → 跳转到另一个 Agent
```

交接工具的返回内容应该告诉模型交接是否成功，但真正的下一步由图路由决定。不要依赖模型看到“成功转移”后自行猜测节点；应使用 `Command.goto` 或明确的条件边。

## 7. State 与消息协议

### 7.1 `MessagesState` 让 Agent 共享对话历史

`MessagesState` 可以理解为：

```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` 会合并新增消息，并按消息 ID 更新已有消息。节点通常只返回本轮新增内容：

```python
return {"messages": [ai_msg, tool_msg]}
```

不要在普通追加场景中返回完整旧列表，否则容易重复历史；也不要把消息对象替换成字符串，否则会丢失 `tool_calls`、角色和调用 ID。

### 7.2 交接时传完整历史还是结构化状态

两种方案各有边界：

| 方案 | 优点 | 风险 |
| --- | --- | --- |
| 完整 `messages` | 实现简单，专家可看到上下文 | 上下文变长，隐私和提示注入风险更大 |
| 结构化字段 | 接口清楚，便于校验和审计 | 需要设计字段、转换和缺失值处理 |
| 摘要 + 关键字段 | 控制上下文长度 | 摘要可能丢失事实，需评估质量 |

可参考持久化笔记中的消息裁剪和摘要策略：改变模型输入视图不等于删除 State；删除消息则必须使用消息 reducer 支持的 `RemoveMessage`。对于工具循环，裁剪时还必须保留一组完整的 `AIMessage.tool_calls` 与对应 `ToolMessage`。

### 7.3 `thread_id`、`user_id` 和 `checkpoint_ns`

- `thread_id`：一条会话线程，隔离同一用户的不同对话历史；
- `user_id`：跨线程业务记忆的命名空间，例如偏好或账户资料；
- `checkpoint_ns`：LangGraph 运行时用于区分父图、子图和执行位置的命名空间。

它们不应互相替代。想让用户在新会话中仍保留酒店偏好，应从 Store 按 `user_id` 搜索；想让当前对话继续看到上一轮消息，应使用同一个 `thread_id` 和 Checkpointer。

## 8. 把 RAG、工具和专家组合起来

多智能体的价值通常在“职责边界”而不是 Agent 数量。一个可扩展的业务图可以是：

```text
用户问题
   ↓
问题分类 / supervisor
   ├─ 知识检索 Agent
   │    ├─ 查询改写
   │    ├─ 混合召回、重排序、上下文压缩
   │    └─ 生成带证据的草稿
   ├─ 工具执行 Agent
   │    ├─ 参数校验
   │    ├─ 受限 API / 数据库调用
   │    └─ 返回结构化结果
   └─ 审核 Agent
        ├─ 检查引用和事实一致性
        └─ 需要时要求原 Agent 重试
```

RAG 笔记中的“先定位失败位置”同样适用于多智能体：

- 检索 Agent 找不到资料，问题在知识库、分块、召回或排序；
- 找到资料但回答增加了上下文没有的事实，问题在生成约束或模型；
- 专家选择错误，问题在路由提示词、工具描述或分类节点；
- 工具结果正确但最终回答错误，问题在消息协议、字段转换或终结节点。

评估时不要只看最终答案。应分别记录：路由是否正确、交接次数、各 Agent 输入输出、工具成功率、检索上下文、最终答案和延迟。Ragas 的 `context_recall`、`context_precision`、`faithfulness`、`answer_relevancy` 可以评估 RAG Agent 的证据链；多智能体还需要补充路由准确率、交接成功率和无效循环率。

## 9. 可观察性：先看事件，再看最终答案

### 9.1 用 `stream()` 判断真实执行路径

```python
for update in graph.stream(
    {"messages": [("user", "我想去加勒比海旅行并找酒店")]},
):
    print(update)
```

一次运行至少要能回答：

1. 首个 Agent 是否被调用；
2. 是否生成了交接工具调用；
3. `ToolMessage` 是否和调用 ID 配对；
4. 下一个 Agent 是否读到了目的地或其它任务字段；
5. 最终是正常结束、达到递归限制，还是因工具错误中止。

`stream()` 输出的是事件或节点更新，不等于完整 State；持久化快照则由 Checkpointer 管理。调试时应同时记录节点名、调用 ID、Agent 名称和耗时，生产日志要做脱敏。

### 9.2 图可视化能看到什么

```python
graph.get_graph().draw_mermaid_png(
    output_file_path="../../../../assets/multi-agent.png"
)
```

静态图可以确认节点是否注册、入口和可能的动态边是否存在，但不能告诉你本次模型实际选择了哪条路径，也不能显示循环了几轮。动态路径必须结合 `stream()` 或 tracing 观察。

`Command[Literal[...]]` 有助于让静态图显示可能目标；如果目标类型过于宽泛，图可能无法完整推断动态边。

## 10. 中断、重试和错误处理

多智能体把单个 Agent 的失败传播到了整个网络，因此需要分层处理：

| 问题 | 处理位置 |
| --- | --- |
| 工具参数错误 | 工具 schema 或自定义工具节点，返回可理解的 ToolMessage |
| 临时网络失败 | 工具节点或节点级 `RetryPolicy`，确认副作用是否可重试 |
| Agent 选错角色 | 路由提示词、工具描述、分类节点和离线评测 |
| 两个 Agent 来回交接 | State 中的阶段/次数/已访问集合，加递归限制 |
| 需要人批准的写操作 | `interrupt()` + Checkpointer + 同一 `thread_id` 恢复 |
| 上下文过长 | 消息裁剪、摘要或把中间结果变成结构化字段 |

`interrupt()` 只负责暂停，Checkpointer 负责保存，`Command(resume=...)` 负责恢复。恢复后节点可能从头重跑，因此写文件、发消息、扣款等副作用必须设计成幂等操作。

不要把异常全部吞掉再返回“处理成功”。错误应包含稳定的错误类别和下一步建议，日志中隐藏密钥、完整授权头、个人信息和未脱敏的工具参数。

## 11. 安全边界

多智能体增加了几个新的攻击面：

1. 一个 Agent 可能把用户输入原样交给另一个 Agent，形成跨角色的提示注入；
2. 交接后的 Agent 可能看到不属于它的敏感历史；
3. 工具调用参数由模型产生，不能当作授权结果；
4. supervisor 的路由决定了哪些工具和数据会被暴露；
5. 多轮交接可能放大单个错误或泄露内容。

建议把“可见上下文、可见工具、可写字段、可跳转节点”都列成白名单。对外部操作增加用户确认或人工审批；对 SQL、URL、文件路径和账户标识做结构化校验；对每次交接记录发起者、目标、原因、消息摘要和结果。

## 12. 离线验证方法

完整示例依赖模型服务，但核心协议可以离线测：

### 12.1 直接测试工具

```python
assert add.invoke({"a": 3, "b": 5}) == 8
assert multiply.invoke({"a": 8, "b": 12}) == 96
```

### 12.2 构造假的 `AIMessage`

用预设的 `AIMessage(tool_calls=[...])` 验证 `ToolNode` 是否能按名称执行工具，并用 `tool_call_id` 建立结果映射。这样不需要 API 凭据，也不依赖正在运行的模型服务。

### 12.3 用假的模型验证交接

至少覆盖以下路径：

- 旅行专家直接回答，不发生交接；
- 旅行专家请求酒店专家，酒店专家正常结束；
- 酒店专家请求返回旅行专家，并在达到上限时停止；
- 工具参数是字典和 JSON 字符串两种格式；
- 工具抛出异常时能返回可识别的错误消息；
- `ToolMessage.tool_call_id` 与对应 `AIMessage` 的调用 ID 一致。

测试不应访问真实 API、Ollama、浏览器或数据库。把模型作为参数传入 `make_agent`，就能用假的 Runnable 替换真实模型；把模型创建、绘图和示例执行放到主入口后，导入模块也不会产生副作用。

## 13. 阅读和扩展现有示例的顺序

建议按下面顺序动手：

1. 先阅读 `如何构建多智能体网络.py`，只关注两个节点、消息历史和 `Command.goto`；
2. 将 `transfer_to_*` 工具改成一个有明确返回值的模拟工具，观察普通工具与交接工具的差异；
3. 阅读 `使用Command进行交接.py`，打印每次 `AIMessage` 的 `tool_calls` 和 `Command` 的目标；
4. 阅读 `使用工具实现交接.py` 的 `make_agent`，先只启用 `add`、`multiply` 两个普通工具；
5. 取消注释多 Agent 版本前，先为 `call_model` 和 `call_tools` 加上假的模型测试；
6. 增加 `handoff_count` 或 `stage` 字段，验证循环和降级出口；
7. 给父图加 Checkpointer，用同一 `thread_id` 测试中断恢复；
8. 最后把一个检索 Agent 和一个工具 Agent 接到 supervisor，再用固定评测集比较路由和回答质量。

## 14. 本章要点

```text
Agent = 角色提示词 + 可见工具 + 消息历史 + 路由规则
多智能体 = 多个职责边界清晰的 Agent + 明确的交接协议
交接工具 = 模型提出“换人”请求，图负责真正跳转
Command.update = 更新 State
Command.goto = 决定下一节点
MessagesState = 用 add_messages 合并消息历史
ToolMessage.tool_call_id = 配对工具请求和工具结果
thread_id = 会话隔离
user_id = 跨线程业务记忆的命名空间
recursion_limit = 图步骤的硬上限
interrupt + checkpointer + Command(resume) = 可恢复的人机协作
stream + tracing = 观察真实路径
```

看到一个多智能体图时，可以依次检查：**每个 Agent 能看到什么 → 能调用哪些工具 → 如何发起交接 → 交接写入哪些 State → 目标节点是否已注册 → 如何结束、恢复和降级**。这比只看最后一句模型回答更能定位问题。
