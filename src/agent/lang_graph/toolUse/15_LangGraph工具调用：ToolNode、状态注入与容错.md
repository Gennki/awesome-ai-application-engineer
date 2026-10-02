# LangGraph 工具调用：ToolNode、状态注入与容错

`toolUse` 目录里的示例从最小的工具绑定开始，逐步扩展到完整的工具调用图、状态和运行配置注入、共享存储、工具更新图状态、错误恢复、节点重试，以及大量工具的动态筛选。

这组代码最好按“**模型提出请求 → 图执行工具 → 结果回到模型 → 必要时继续循环**”来理解。模型只生成结构化的 `tool_calls`，真正调用 Python 函数的是 `ToolNode` 或自定义工具节点；状态、配置和 Store 则决定工具在执行时能看到哪些上下文。

> 本文按当前文件内容整理。目录中的脚本大多会在导入时创建模型、绘制图片或直接发起请求，阅读和离线验证时应优先把示例逻辑提取成函数，并在 `if __name__ == "__main__":` 中运行。

## 1. 文件地图与推荐阅读顺序

| 文件 | 主要问题 | 需要记住的 API |
| --- | --- | --- |
| [模型调用.py](模型调用.py) | 模型拿到工具说明后会返回什么 | `@tool`、`bind_tools()` |
| [LangGraph_ToolNode.py](LangGraph_ToolNode.py) | 怎样把模型和工具接成循环 | `MessagesState`、`ToolNode`、条件边 |
| [如何将图状态传递给工具.py](如何将图状态传递给工具.py) | 工具怎样读取当前图状态 | `AgentState`、`InjectedState` |
| [如何传递配置给工具.py](如何传递配置给工具.py) | 工具怎样读取每次调用的配置 | `RunnableConfig` |
| [传递共享存储.py](传递共享存储.py) | 工具怎样读取跨调用、按用户隔离的数据 | `InjectedStore`、`BaseStore`、`InMemoryStore` |
| [如何从工具更新图状态.py](如何从工具更新图状态.py) | 工具怎样写回自定义状态并补充消息 | `Command(update=...)`、`InjectedToolCallId` |
| [异常ToolNode.py](异常ToolNode.py) | 工具抛出异常时如何转成工具结果 | `ToolNode(handle_tool_errors=True)` |
| [自定义策略.py](自定义策略.py) | 如何删除失败尝试并切换备用模型 | `ToolMessage`、`RemoveMessage`、条件边 |
| [示例9_如何添加节点重试策略.py](示例9_如何添加节点重试策略.py) | 节点失败后如何自动重试 | `RetryPolicy` |
| [如何处理大量工具.py](如何处理大量工具.py) | 工具很多时如何先检索再绑定 | `InMemoryVectorStore`、动态 `bind_tools()` |

推荐顺序是：先读“模型调用”和“LangGraph_ToolNode”，再读三种注入方式，最后读错误处理、重试和大量工具。这样可以把每个高级 API 放回已经理解的工具调用闭环中。

## 2. 先分清四个角色

工具调用经常被误解成“模型直接执行了 Python 函数”。实际至少有四个角色：

| 角色 | 作用 | 目录中的实现 |
| --- | --- | --- |
| 工具定义 | 描述名称、参数和用途，并提供实际函数 | `@tool`、`StructuredTool.from_function()` |
| 模型绑定 | 把工具的名称、描述和参数 schema 告诉模型 | `model.bind_tools(tools)` |
| 工具执行节点 | 读取 `AIMessage.tool_calls`，找到并调用工具 | `ToolNode(tools)` |
| 图路由 | 决定执行工具后是否回到模型，或结束 | `should_continue()`、`tools_condition` |

可以把一轮调用简化成下面的消息链：

```text
HumanMessage：用户问题
      ↓
AIMessage：tool_calls = [{name, args, id}]
      ↓
ToolNode：按 name 找到工具并执行 args
      ↓
ToolMessage：工具返回值 + tool_call_id
      ↓
AIMessage：模型根据工具结果回答，或再次请求工具
```

`bind_tools()` 只完成第二步，不会执行函数；`ToolNode` 只负责执行请求，不负责决定模型应该问什么。只有把两者放进一张有回边的图，才能得到完整的 Agent 工具循环。

## 3. 从 `@tool` 到 `tool_calls`

### 3.1 `@tool` 生成给模型看的工具说明

[模型调用.py](模型调用.py) 中的工具是：

```python
from langchain_core.tools import tool


@tool
def get_weather(location: str):
    """获取当前天气"""
    if location.lower() in ["SH", "上海"]:
        return "气温23度，有雾。"
    return "气温30度，阳光明媚。"
```

`@tool` 会把普通函数包装成工具对象。函数名、参数名、类型注解和 docstring 共同组成工具 schema：

- 名称帮助模型选择工具；
- 类型注解帮助模型生成结构化参数；
- docstring 说明工具适合什么时候使用；
- 函数体仍然由 Python 执行。

工具描述必须符合真实行为。如果函数返回的是固定演示数据，就应该把它当作占位实现，不要在说明中声称它访问了实时天气服务。

### 3.2 `bind_tools()` 返回的是请求，不是执行结果

```python
tools = [get_weather, get_coolest_cities]
model_with_tools = getModel().bind_tools(tools)
response = model_with_tools.invoke("上海的天气怎么样？")
print(response.tool_calls)
```

模型响应可能类似：

```python
[
    {
        "name": "get_weather",
        "args": {"location": "上海"},
        "id": "call_...",
        "type": "tool_call",
    }
]
```

这一轮通常还没有真正运行 `get_weather()`。`content` 可能为空，`tool_calls` 才是模型提出的结构化请求。程序如果只使用 `bind_tools()`，就必须自己完成以下工作：

1. 检查工具名是否在允许列表中；
2. 校验参数并执行对应函数；
3. 创建带有相同 `tool_call_id` 的 `ToolMessage`；
4. 把原消息、工具请求和工具结果一起发回模型。

目录中的后续示例把第 2、3 步交给 `ToolNode`，把第 4 步交给图的回边。

## 4. `ToolNode` 闭环：模型、工具和条件边

[LangGraph_ToolNode.py](LangGraph_ToolNode.py) 建立的是最典型的两节点图：

```text
START → agent(call_model)
              │
              ├─ 有 tool_calls → tools(ToolNode) → agent
              └─ 没有 tool_calls → END
```

核心代码可以缩写为：

```python
tool_node = ToolNode(tools)
model_with_tools = getModel().bind_tools(tools)


def call_model(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

### 4.1 为什么要使用 `MessagesState`

`MessagesState` 为 `messages` 配置了消息 reducer。节点只需返回本次新增的消息：

```python
return {"messages": [response]}
```

LangGraph 会把它合并到已有历史中。一次成功的单工具调用通常包含：

| 顺序 | 消息 | 作用 |
| ---: | --- | --- |
| 1 | `HumanMessage` | 用户问题 |
| 2 | `AIMessage` | 模型提出工具调用 |
| 3 | `ToolMessage` | `ToolNode` 执行后的结果 |
| 4 | `AIMessage` | 模型读取结果后生成回答 |

必须保留完整的 `AIMessage`，不能只返回 `response.content`。否则后续节点看不到 `tool_calls`，也无法路由到工具节点。

### 4.2 `should_continue()` 只负责路由

`should_continue()` 不执行工具，也不生成回答，它只读取最新消息并返回下一个节点名。`add_conditional_edges()` 的列表表示“可能的目标”，不是每个目标都执行；具体走向仍由条件函数的返回值决定。

LangGraph 也提供了等价的 `tools_condition`：

```python
from langgraph.prebuilt import tools_condition

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END},
)
```

如果把工具节点改名为 `calculator`，映射也必须改成 `{"tools": "calculator", END: END}`。

### 4.3 一个 `ToolNode` 可以执行多个工具

`tools` 列表中有两个工具，并不意味着每次都会执行两个。模型可能只请求 `get_weather`，也可能在一条 `AIMessage` 中同时请求多个工具。`ToolNode` 会按 `name` 分发请求，并为每个请求生成对应的 `ToolMessage`。

静态图中只会看到一个 `tools` 节点，因为两个 Python 工具是注册在同一个 `ToolNode` 内部的。要知道本次到底执行了哪些工具，应查看 `tool_calls`、`ToolMessage.name` 或流式状态更新。

### 4.4 `模型调用.py` 与 `LangGraph_ToolNode.py` 的区别

`模型调用.py` 只执行：

```python
model_with_tools.invoke("上海的天气怎么样？")
```

所以它用于观察模型是否提出工具请求；文件中创建的 `tools_node` 并没有被调用。

`LangGraph_ToolNode.py` 才把模型响应交给 `ToolNode`，并通过 `tools → agent` 的回边让模型读取工具结果。学习时可以先运行第一种思路理解消息，再用第二种思路理解图。

## 5. 把图状态注入工具：`InjectedState`

[如何将图状态传递给工具.py](如何将图状态传递给工具.py) 定义了一个额外状态字段：

```python
class State(AgentState):
    docs: list[str]
```

工具需要用户问题和当前状态中的文档：

```python
@tool
def get_context(
    question: str,
    state: Annotated[dict, InjectedState],
):
    """获取回答问题的相关背景"""
    return "\n\n".join(state["docs"])
```

这里有两个不同来源的参数：

| 参数 | 谁提供 | 是否应该让模型填写 |
| --- | --- | --- |
| `question` | 模型根据用户问题生成 | 是 |
| `state` | `ToolNode` 从当前图状态注入 | 否 |

`InjectedState` 的意义是：工具可以读取整份图状态，但 `state` 不会作为普通工具参数暴露给模型。模型只需要决定 `question`，不需要复制 `docs`。

### 5.1 两种 schema 要区分

示例打印了两个 schema：

```python
get_context.get_input_schema().model_json_schema()
get_context.tool_call_schema.model_json_schema()
```

可以这样理解：

- `get_input_schema()` 描述工具对象在 Python 中接收的输入形状；
- `tool_call_schema` 描述发给模型的工具调用参数。

注入参数在执行时必须存在，但不应出现在模型需要填写的参数中。调试工具调用失败时，应先检查模型看到的 `tool_call_schema`，再检查 `InjectedState` 是否在实际图状态中存在。

### 5.2 状态字段必须先进入图输入

示例调用图时同时传入 `messages` 和 `docs`：

```python
inputs = {
    "messages": [{"type": "user", "content": "关于FooBar有什么最新消息"}],
    "docs": [
        "FooBar公司刚刚筹集了10亿美元",
        "FoBar公司成立于2019年",
    ],
}
```

如果遗漏 `docs`，工具虽然能被模型调用，但在 `state["docs"]` 处会失败。`InjectedState` 读取的是当前图状态，不是某个全局变量，也不会自动从 Checkpointer 中推断出业务字段。

## 6. 把运行配置注入工具：`RunnableConfig`

[如何传递配置给工具.py](如何传递配置给工具.py) 用配置中的用户标识维护“用户喜欢的宠物”：

```python
@tool
def update_favorite_pets(
    pets: list[str],
    config: RunnableConfig,
) -> None:
    user_id = config.get("configurable", {}).get("user_id")
    user_to_pets[user_id] = pets
```

配置从调用方传入：

```python
config = {
    "configurable": {
        "thread_id": "thread-123",
        "user_id": "user-1",
    }
}
graph.stream(inputs, config, stream_mode="values")
```

### 6.1 `thread_id` 和 `user_id` 不是一回事

- `thread_id` 是 Checkpointer 的会话键，决定恢复哪一份对话状态；
- `user_id` 是业务配置，工具用它选择用户数据；
- 同一个用户可以有多个线程；
- 不同用户不能共用一个业务数据键，除非这是明确的产品设计。

当前 [如何传递配置给工具.py](如何传递配置给工具.py) 的三个测试只传了 `thread_id`，没有传 `user_id`。因此 `user_id` 会取到 `None`，三个操作实际都使用同一个字典键。要演示真正的用户隔离，应改为：

```python
config = {
    "configurable": {
        "thread_id": "thread-123",
        "user_id": "user-1",
    }
}
```

这也是阅读配置注入示例时最容易忽略的地方：**配置不会自动写进 State，也不会因为存在 `thread_id` 就自动产生 `user_id`。**

### 6.2 注入参数不应成为模型的业务参数

`config: RunnableConfig` 是运行时上下文，由框架注入。工具 docstring 应描述模型需要决定的业务参数，例如 `pets`；不要让模型尝试生成 `config`、`thread_id` 或用户身份。

另外，示例中的 `list_favorite_pets()` 标注为 `-> None`，但实际返回逗号拼接的字符串。生产代码应把返回注解改成 `-> str`，使 schema 和真实行为一致。

### 6.3 配置适合放什么

适合注入配置的内容包括：

- 用户或租户标识；
- 请求追踪 ID；
- 当前环境或区域；
- 运行时选择的模型名称；
- 权限校验所需的服务端上下文。

不要把密钥、完整授权头或未经筛选的内部对象交给模型。配置是程序上下文，工具内部仍要做权限检查。

## 7. 注入共享存储：`InjectedStore`

配置适合传递“当前请求是谁”，Store 适合保存可跨调用复用的业务数据。[传递共享存储.py](传递共享存储.py) 使用 `InMemoryStore`，并按用户划分命名空间：

```python
doc_store = InMemoryStore()
doc_store.put(
    ("documents", "1"),
    "doc_0",
    {"doc": "FooBar公司刚刚筹集了10亿美元！"},
)
doc_store.put(
    ("documents", "2"),
    "doc_1",
    {"doc": "FooBar公司成立于2019年"},
)
```

工具通过两个注入参数读取它：

```python
@tool
def get_context(
    question: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
):
    user_id = config.get("configurable", {}).get("user_id")
    items = store.search(("documents", user_id))
    return "\n\n".join(item.value["doc"] for item in items)
```

### 7.1 State、Checkpointer 和 Store 的边界

| 机制 | 保存什么 | 典型用途 |
| --- | --- | --- |
| State | 当前图运行中的字段 | `messages`、本轮 `docs`、`user_info` |
| Checkpointer | 某个 `thread_id` 的状态快照 | 多轮对话、暂停后恢复 |
| Store | 跨线程、按命名空间访问的业务数据 | 用户资料、知识片段、长期记忆 |
| RunnableConfig | 本次调用的运行上下文 | `user_id`、追踪信息、运行参数 |

`thread_id` 让 Checkpointer 找到会话；`user_id` 让工具找到 Store 中的命名空间。两者可以相同，也可以不同，但应根据业务语义分别设计。

### 7.2 创建 Agent 时必须传入 Store

示例显式写出：

```python
graph = create_agent(
    model,
    [get_context],
    checkpointer=MemorySaver(),
    store=doc_store,
)
```

只在工具签名里写 `InjectedStore()`，而不把 `store=doc_store` 交给 Agent，工具执行时没有可注入的存储。`ToolNode` 会负责注入 `RunnableConfig` 和 Store，但前提是图运行时确实拥有这些依赖。

当前示例的返回注解写成了 `Tuple[str, List[Document]]`，实际只返回字符串。应按真实返回值改成 `-> str`，或者真的返回文档对象，避免类型和工具 schema 误导调用方。

## 8. 工具更新图状态：`Command(update=...)`

普通工具返回一个字符串时，`ToolNode` 通常只生成 `ToolMessage`。如果工具还要更新图中的自定义字段，可以返回 `Command`。[如何从工具更新图状态.py](如何从工具更新图状态.py) 扩展了 `AgentState`：

```python
class State(AgentState):
    user_info: dict[str, Any]
```

工具查到用户资料后返回：

```python
return Command(
    update={
        "user_info": user_info,
        "messages": [
            ToolMessage(
                "成功查询用户信息",
                tool_call_id=tool_call_id,
            ),
            SystemMessage(content=f"用户信息：{user_info}"),
        ],
    }
)
```

这里同时更新了两类数据：

- `user_info`：供后续节点或模型使用的业务状态；
- `messages`：让消息链保留工具成功执行的证据和补充上下文。

### 8.1 `InjectedToolCallId` 的作用

```python
tool_call_id: Annotated[str, InjectedToolCallId]
```

工具不需要让模型填写调用 ID。框架会从当前的 `AIMessage.tool_calls` 中注入它。工具返回 `ToolMessage` 时必须使用同一个 ID，模型才能把结果和自己的那次请求对应起来。

### 8.2 `Command` 与 `ToolMessage` 的关系

`Command(update=...)` 是“工具执行后如何更新图状态”的封装；`ToolMessage` 是消息历史中的工具结果。两者职责不同，但通常会放在同一个 `update` 中：一个更新业务字段，一个补齐消息链。

这和 `Command(resume=...)` 不同。后者是调用方在 `interrupt()` 暂停后传回的恢复命令；本示例的 `Command` 是工具节点返回的状态更新，不涉及人工中断。

## 9. 工具异常：内置处理和自定义恢复

### 9.1 `ToolNode(handle_tool_errors=True)`

[异常ToolNode.py](异常ToolNode.py) 的工具会针对不同输入抛出 `ValueError`：

```python
tool_node = ToolNode([get_weather], handle_tool_errors=True)
```

开启后，工具异常可以被包装成工具错误消息，再交回模型处理，而不是立即让整个图调用失败。图结构仍然是：

```text
agent → tools → agent
```

例如用户询问“长沙的天气”，工具抛出“无效输入”，模型可能读取这条错误消息后请求更合适的输入或向用户解释失败原因。

`handle_tool_errors=True` 不是“模型一定能修复任何错误”。工具名不存在、状态缺失、依赖初始化失败，以及不同 LangGraph 版本对异常的处理方式，都需要单独验证。生产应用还应记录错误类型、调用 ID 和重试次数，但不要把内部堆栈或凭据泄露给用户。

### 9.2 自定义工具节点和备用模型

[自定义策略.py](自定义策略.py) 没有直接使用 `ToolNode`，而是手动遍历最后一条消息的 `tool_calls`：

```python
for tool_call in last_message.tool_calls:
    try:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        output_messages.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    except Exception as error:
        output_messages.append(
            ToolMessage(
                content="",
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                additional_kwargs={"error": error},
            )
        )
```

自定义流程有四个节点：

```text
agent（基础模型）
  ├─ 无工具请求 → END
  └─ 有工具请求 → tools
                    ├─ 成功 → agent
                    └─ 失败 → remove_failed_tool_call_attempt
                                      ↓
                                fallback_agent（备用模型）
                                      ↓
                                    tools
```

### 9.3 为什么要删除失败的尝试

`remove_failed_tool_call_attempt()` 找到最近的 `AIMessage`，并用 `RemoveMessage` 删除从这条消息开始的消息。这样备用模型不会把“失败的工具请求 + 错误结果”当成已经确认的历史，能够重新生成一次工具调用。

这个节点体现了一个重要原则：**失败恢复不一定是继续追加消息，有时要先修剪错误分支。** 如果保留失败调用，备用模型可能重复引用错误参数，或者把错误文本当成真实业务结果。

### 9.4 自定义策略的工程注意事项

- `tools_by_name` 应覆盖所有允许的工具，并对未知工具名给出明确错误；
- `additional_kwargs["error"]` 当前保存的是异常对象，若要持久化、序列化或跨进程传输，应保存错误类型和安全的字符串摘要；
- `json.dumps(tool_result)` 只适合可 JSON 序列化的返回值，复杂对象应先转换；
- 删除消息后要确认 reducer 支持 `RemoveMessage`，并保留合法的调用 ID 配对；
- 备用模型仍可能失败，应该设置递归限制和最终兜底分支。

## 10. 节点重试：`RetryPolicy` 与工具错误处理不是一层

[示例9_如何添加节点重试策略.py](示例9_如何添加节点重试策略.py) 为 `model` 节点配置了：

```python
builder.add_node(
    "model",
    call_model,
    retry=RetryPolicy(max_attempts=5),
)
```

这表示 `call_model` 抛出符合策略的异常时，LangGraph 可以重新执行这个节点，最多尝试五次。它和 `ToolNode(handle_tool_errors=True)` 的层级不同：

| 机制 | 处理对象 | 失败后通常发生什么 |
| --- | --- | --- |
| `handle_tool_errors` | 工具函数执行异常 | 生成错误工具消息，继续图流程 |
| `RetryPolicy` | 节点执行抛出的异常 | 重新调用整个节点 |
| 自定义 fallback | 业务判定出的失败消息 | 清理消息并切换节点/模型 |

示例流程是：

```text
START → model（生成 SQL）→ query_database（执行 SQL）→ END
```

当前 `RetryPolicy` 只加在 `model` 节点上。被注释的代码展示了也可以针对数据库异常配置重试，但要先导入具体异常类型，并确认数据库操作是否适合重复执行。

### 10.1 重试节点必须考虑副作用

纯模型调用、幂等查询通常更适合自动重试。发送邮件、扣款、写数据库等副作用操作重复执行可能造成重复结果，应使用幂等键、事务或人工确认。重试不是超时保护，也不会自动修正模型生成的错误 SQL。

### 10.2 这个 SQL 示例的学习重点

`query_database()` 从模型消息中提取 `SELECT ...;`，去除常见 Markdown 代码围栏，再执行 SQLite 查询。它说明了一个安全边界：模型生成的 SQL 仍然必须经过程序校验，不能因为节点配置了重试就把任意 SQL 直接交给数据库。

源码默认读取 `state['messages'][1]` 作为模型输出，因此它依赖固定消息顺序。实际应用应按消息类型或节点输出取值，并限制只读语句、允许访问的表和返回行数。

## 11. 工具很多时：先检索工具，再动态绑定

当工具数量从两个增长到几百个时，把完整工具 schema 每次都交给模型会增加上下文长度，也会让工具选择变得困难。[如何处理大量工具.py](如何处理大量工具.py) 使用“工具注册表 + 工具描述向量检索”的两阶段流程：

```text
用户问题
   ↓
select_tools：在工具描述向量库中检索相关工具 ID
   ↓
agent：只把选中的工具绑定给模型
   ↓
ToolNode：执行模型请求的工具
   ↓
agent：读取工具结果并结束或继续
```

### 11.1 工具注册表和描述文档

每个工具放在注册表中，UUID 作为稳定内部 ID：

```python
tool_registry = {
    str(uuid.uuid4()): create_tool(company)
    for company in s_and_p_500_companies
}
```

随后把工具描述转换成 `Document`，把注册表 ID 放进 `Document.id`：

```python
tool_documents = [
    Document(
        page_content=tool.description,
        id=tool_id,
        metadata={"tool_name": tool.name},
    )
    for tool_id, tool in tool_registry.items()
]
```

检索节点只更新 `selected_tools`：

```python
def select_tools(state: State):
    query = state["messages"][-1].content
    documents = vector_store.similarity_search(query)
    return {"selected_tools": [document.id for document in documents]}
```

模型节点再根据这些 ID 动态绑定：

```python
current_tools = [tool_registry[id] for id in state["selected_tools"]]
llm_with_selected_tools = model.bind_tools(current_tools)
return {"messages": [llm_with_selected_tools.invoke(state["messages"])]}
```

### 11.2 为什么 `ToolNode` 仍然注册全部工具

示例中：

```python
tools = list(tool_registry.values())
tool_node = ToolNode(tools=tools)
```

模型每轮只看到检索出来的工具，但 `ToolNode` 维护着完整注册表。这样工具请求到达执行节点时仍能按名称找到实际函数。生产代码可以进一步限制执行范围，检查请求的工具名是否属于本轮 `selected_tools`，防止模型调用未被检索的工具。

### 11.3 动态绑定的边界

- 检索结果为空时要提供无工具的模型分支或明确提示；
- 检索返回过多工具时要设置 `k`、分数阈值或二次排序；
- 工具描述质量直接影响召回，描述应包含能力、参数和适用场景；
- 向量检索失败不能让整个 Agent 无限制重试；
- `getEmbedding()` 可能访问外部 embedding 服务，离线运行前要准备替代实现；
- 示例中的 `document_ids` 没有参与后续流程，可以删除或用于校验入库结果。

当前 `create_tool()` 的返回注解写成了 `dict`，实际返回的是 `StructuredTool`，建议修正为 `-> StructuredTool`。工具名称清洗规则也应单独测试，确保不同公司名不会生成重复或非法名称。

## 12. 三种“上下文注入”的选择

目录中出现了三种不同的上下文来源，可以按数据生命周期选择：

| 问题 | 推荐机制 | 示例 |
| --- | --- | --- |
| 本轮图里已有的字段，工具临时读取 | `InjectedState` | 读取 `docs` |
| 本次调用的用户、租户或运行参数 | `RunnableConfig` | 读取 `user_id` |
| 跨线程保存的长期业务数据 | `InjectedStore` | 按用户命名空间检索文档 |

一个工具可以同时使用三者：配置提供 `user_id`，Store 根据它查询长期数据，State 再携带本轮已筛选的文档。关键是保持边界清楚，不要把长期数据复制进每次模型请求，也不要把权限信息交给模型自行决定。

## 13. 运行与排查清单

### 13.1 运行前

1. 确认 `.env` 中的模型和 embedding 配置已加载；
2. 确认当前模型支持 tool calling；
3. 确认 Mermaid/Graphviz 绘图依赖和输出路径可用；
4. 对有副作用的工具先改成模拟函数；
5. 避免通过导入脚本触发文件末尾的模型请求。

### 13.2 模型没有调用工具

按这个顺序检查：

1. `response.tool_calls` 是否为空；
2. docstring 是否明确说明工具适用场景；
3. 参数是否有正确的类型注解；
4. 模型是否支持工具调用；
5. 用户问题是否真的需要工具；
6. 是否误把 `content` 当成工具执行结果。

### 13.3 工具节点报错

1. `tool_call["name"]` 是否和注册工具名称一致；
2. `args` 是否符合工具 schema；
3. `InjectedState` 对应字段是否存在；
4. `RunnableConfig` 中是否真的传入了 `user_id`；
5. 使用 `InjectedStore` 时是否在 `create_agent()` 传入了 `store`；
6. `ToolMessage.tool_call_id` 是否和请求的 ID 相同；
7. 工具返回值是否可序列化、是否与返回注解一致。

### 13.4 图一直循环

工具循环必须最终产生一条没有 `tool_calls` 的 AI 消息。可以检查：

- 工具结果是否真的回到了 `messages`；
- 模型是否看到了 `ToolMessage`；
- 是否错误地丢弃了完整的 `AIMessage`；
- 工具是否总是返回错误，导致模型不断重试；
- 是否缺少 `recursion_limit`；
- 自定义 fallback 是否有最终结束分支。

可以在调用时设置上限：

```python
result = app.invoke(
    {"messages": [{"role": "user", "content": "查询天气"}]},
    config={"recursion_limit": 10},
)
```

`recursion_limit` 限制的是图的执行步数，不是最多调用多少次工具，也不是模型 Token 上限。超过限制时通常会抛出 `GraphRecursionError`，不会自动生成最终答案。

## 14. 一张总图

```mermaid
flowchart TD
    START([START]) --> MODEL[模型节点\nbind_tools]
    MODEL -->|无 tool_calls| END([END])
    MODEL -->|有 tool_calls| TOOL[ToolNode\n执行工具]
    TOOL --> MODEL

    TOOL -.-> STATE[InjectedState\n读取图状态]
    TOOL -.-> CONFIG[RunnableConfig\n读取本次配置]
    TOOL -.-> STORE[InjectedStore\n读取共享存储]
    TOOL -.-> UPDATE[Command(update=...)\n写回图状态]

    TOOL --> ERROR{工具异常}
    ERROR -->|内置错误消息| MODEL
    ERROR -->|自定义策略| CLEAN[删除失败尝试]
    CLEAN --> FALLBACK[备用模型]
    FALLBACK --> TOOL

    SELECT[工具描述检索] --> MODEL
    RETRY[RetryPolicy\n节点级重试] -.-> MODEL
```

图中的 `InjectedState`、`RunnableConfig` 和 `InjectedStore` 是工具执行时的输入来源；`Command(update=...)` 是工具向图写回状态的方式；`RetryPolicy` 作用于节点异常，和工具错误消息属于不同层次。

## 15. 练习建议

1. 先把 [模型调用.py](模型调用.py) 改成打印 `response.tool_calls`，确认模型只提出请求。
2. 在 [LangGraph_ToolNode.py](LangGraph_ToolNode.py) 中打印每条消息的类型、工具名和调用 ID，观察四段消息链。
3. 给 [如何将图状态传递给工具.py](如何将图状态传递给工具.py) 增加 `docs=[]` 和缺少 `docs` 两个测试，比较正常结果和异常位置。
4. 修正 [如何传递配置给工具.py](如何传递配置给工具.py) 的配置，分别使用 `user-1`、`user-2`，验证宠物数据不会串用户。
5. 在共享 Store 示例中增加第三个命名空间，并比较不同 `thread_id` 使用同一个 `user_id` 时的结果。
6. 把 [异常ToolNode.py](异常ToolNode.py) 的 `handle_tool_errors` 改成自定义错误消息，观察模型收到的 `ToolMessage`。
7. 给自定义策略增加“备用模型也失败”的结束分支和最大重试计数。
8. 把大量工具示例的 `similarity_search()` 改成显式的 `k` 和分数阈值，观察选中工具数量变化。

读完这组示例后，可以回看 [Tool 与 Function Calling](../../../tool/04_Tool与FunctionCall.md)、[初识 Graph](../base/09_初识Graph：State、Node与Edge.md) 和 [LangGraph 导读](../../08_LangGraph导读.md)，把“工具调用协议”“图状态合并”和“Agent 内部图”放在同一个执行模型中理解。
