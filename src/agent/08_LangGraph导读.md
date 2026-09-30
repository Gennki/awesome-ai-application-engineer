# LangGraph 导读：先把 Agent 看成一张会保存状态的流程图

前面已经学过 RAG、LangChain、Tool 和 Agent，这时候不需要立即记住 LangGraph 的所有 API。先抓住一个核心：

> LangGraph 把 Agent 的执行过程表示成“状态 + 节点 + 边”，并可以在多次调用之间保存状态。

本篇只帮助看懂 [lc_mem_agent.py](lc_mem_agent.py) 的整体结构。条件分支、工具节点、人工审批和持久化等细节先不展开。

## 1. 从 Chain 走到 Graph

LCEL 中的线性 chain 很像一条流水线：

```text
Prompt -> Model -> Parser
```

每次都按预定顺序向前执行。但 Agent 往往不是一条直线：模型可能直接回答，也可能先调用工具，根据工具结果再决定是否继续。

```text
             +----------+
             |          v
用户输入 -> 模型判断 -> 调用工具
             |              |
             v              |
          生成回答 <-------+
```

当流程出现分支、循环、重试、人工确认或多轮状态时，用“图”表达会更清楚。

用已经熟悉的 RAG 来想，一个更复杂的流程可能是：

```text
START -> 检索 -> 判断资料是否足够
                    | 足够   -> 生成回答 -> END
                    | 不足够 -> 改写问题 -> 重新检索
```

LangGraph 的价值不是“让模型更聪明”，而是让程序更明确地管理这些步骤和状态。

## 2. 先记住四个角色

| 概念 | 在图里是什么 | 在当前 demo 中对应什么 |
|---|---|---|
| State | 整个流程共享的数据 | `AgentState` 中的 `messages` |
| Node | 读取状态并返回更新的一步 | `call_model` |
| Edge | 规定下一步去哪个节点 | `agent -> END` |
| Checkpointer | 按会话保存图的状态快照 | `InMemorySaver` |

可以把 State 想成一本公用笔记本：每个 Node 都能看到当前内容，完成自己的工作后再写回一部分。Edge 决定接下来把笔记本交给谁。

## 3. 当前 demo 的图很简单

[lc_mem_agent.py](lc_mem_agent.py) 构建的外层图只有一个节点：

```text
START -> agent(call_model) -> END
```

对应代码是：

```python
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
```

- `StateGraph(AgentState)` 声明这张图使用哪种状态结构。
- `add_node()` 把 Python 函数注册成节点。
- `set_entry_point()` 指定第一个执行的节点。
- `add_edge()` 指定节点执行完后去哪里。
- `END` 是图的结束标记，不是需要自己实现的函数。

这张图还没有展示 LangGraph 最擅长的分支和循环，它的作用是让我们先看清最小结构。

## 4. State：为什么 `messages` 不会被覆盖

状态定义如下：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

`TypedDict` 描述状态中有哪些字段。这里只有 `messages`，用来保存对话消息。

`Annotated[list, add_messages]` 不只是类型标注，它还告诉 LangGraph：当节点返回新消息时，用 `add_messages` 将它们合并进已有消息。它通常会追加新消息；如果新旧消息的 id 相同，则用新消息更新旧消息。这种决定“新值如何写入旧状态”的函数叫 **reducer**。

可以先简化理解为：

```text
原 messages + 新 messages -> 更新后的 messages
```

如果没有合适的 reducer，节点返回的新列表默认可能直接替换旧值，无法得到期望的对话累积效果。

## 5. Node：接收状态，返回局部更新

```python
def call_model(state: AgentState):
    response = agent.invoke({"messages": state["messages"]})
    return {"messages": [response["messages"][-1]]}
```

这个节点做了三件事：

1. 从 State 读取目前的全部对话消息。
2. 把消息交给 `agent.invoke()`，让 Agent 完成模型判断和工具调用。
3. 只返回最后一条 AI 消息，由 `add_messages` 将它追加到外层图的 State。

节点返回的是“状态更新”，不是完整 State 的副本。未来 State 有 `messages`、`documents`、`retry_count` 等多个字段时，某个节点可以只返回它负责更新的部分。

## 6. `compile()` 之后才是可运行的图

```python
checkpointer = InMemorySaver()
graph = graph.compile(checkpointer=checkpointer)
```

在 `compile()` 之前，`graph` 只是一份流程定义。编译会检查结构，并返回可以调用的对象。编译后的图也遵循 LangChain 熟悉的 Runnable 接口，因此可以继续使用 `invoke()`、`stream()` 等方法。

`checkpointer` 是可选的。此处传入 `InMemorySaver` 后，图才能把每次执行后的 State 快照保存起来。

## 7. 记忆是怎样在多次 `invoke()` 之间生效的

demo 中四次调用都使用同一份配置：

```python
config = {"configurable": {"thread_id": "user_123"}}
```

`thread_id` 可以理解为“哪一本会话笔记”的索引。它不是放在 State 中给模型阅读的普通消息，而是交给 checkpointer 用来找到对应状态。

前两次执行可以简化成：

```text
第 1 次：[]
          + Human("我叫张三")
          + AI("好的……")
          -> 保存 [Human1, AI1]

第 2 次：恢复 [Human1, AI1]
          + Human("现在几点")
          + AI("现在是……")
          -> 保存 [Human1, AI1, Human2, AI2]
```

因此第三次问“我叫什么名字”时，模型再次收到的 State 里仍然有第一轮对话。

这里要区分三个概念：

- **State** 定义要在图中传递哪些数据。
- **Checkpointer** 负责在多次调用之间保存和恢复 State。
- **`thread_id`** 决定这次调用读写哪一份状态。

只定义 State 不等于已经拥有跨调用记忆；还需要 checkpointer 和稳定的 `thread_id`。

### 换一个 `thread_id` 会怎样

```python
another_config = {"configurable": {"thread_id": "user_456"}}
```

新 id 会对应一份新状态，因此它不会知道 `user_123` 的对话内容。真实应用中要为不同会话分配不同 id，避免串话。

`thread_id` 只是状态隔离键，不是用户身份认证或权限检查。不能让客户端任意猜测并访问其他人的 id。

### `InMemorySaver` 能记多久

`InMemorySaver` 只把数据放在当前 Python 进程的内存中：

- 适合本地 demo 和测试；
- 程序重启后记忆消失；
- 多进程部署时不能自动共享；
- 生产环境通常需要换成数据库等持久化 checkpointer。

## 8. 这个 demo 其实有两层图

这是理解当前代码最关键的一点。

```python
agent = create_agent(
    model=model,
    tools=[get_current_time, calculator],
    system_prompt="...",
)
```

LangChain 1.x 的 `create_agent()` 底层本身就使用 LangGraph，会在内部管理“模型 -> 工具 -> 再次调用模型”的循环。外层又手动创建了一个 `StateGraph`：

```text
外层 StateGraph
START -> call_model ------------------------------------> END
             |
             v
       create_agent 内部图
       模型 -> 是否调用工具？ -> 工具 -> 模型 -> ……
```

所以，`get_current_time` 和 `calculator` 并不是外层图中显式注册的 Node，而是由内层 `agent` 负责调用。

这种写法用来建立概念没有问题：外层图演示 State、Node、Edge 和 Checkpointer，内层 Agent 继续处理已经熟悉的工具调用。但它不表示任何 Agent 都必须再包一层 `StateGraph`。

对这个只需要工具和短期记忆的简单场景，也可以直接把 checkpointer 传给 `create_agent()`。当需要自己控制多个业务节点、条件路由和循环时，再显式编排 `StateGraph` 更有价值。

## 9. 当前 demo 有意做了简化

`call_model()` 只把内层 Agent 的最后一条回答写回外层 State：

```python
return {"messages": [response["messages"][-1]]}
```

因此外层会话历史主要保留“用户消息 + 最终 AI 回答”，而不保留内层 Agent 的完整工具调用过程。例如调用计算器时，中间的 Tool Call 和 ToolMessage 不会被追加到外层 State。

对于“记住我叫什么”的入门 demo，这种简化足够。如果未来要回放、审计或继续利用完整的工具消息，就需要重新设计写回 State 的内容，或者直接让内层 Agent 使用 checkpointer。

另外，demo 的 `calculator` 即使关闭了 `__builtins__`，仍不应当作通用的生产级表达式执行器。真实应用应使用只允许数字和指定运算符的解析方案。

文件中的四次 `invoke()` 位于模块顶层，因此导入这个文件也会立即发起模型请求。这适合直接运行的 demo；如果以后要在其它模块中复用，应将执行入口放入 `main()` 并使用 `if __name__ == "__main__":` 保护。

## 10. 初学时容易混淆的几件事

- **State 不等于持久化记忆。** State 是数据结构，checkpointer 才负责跨调用保存它。
- **Node 不等于 LLM。** Node 可以调用模型，也可以是普通 Python 函数、检索、校验或人工审批步骤。
- **Edge 不一定是固定顺序。** 本篇只使用普通边；后续的条件边可以根据 State 选择不同下一步。
- **Graph 不会自动产生业务流程。** 节点职责、状态字段、路由条件和结束条件仍需要程序员设计。
- **`thread_id` 不等于登录用户 id。** 它是 checkpointer 的状态隔离键，鉴权应由应用的安全层负责。
- **记忆不代表同一次图运行一直没结束。** 每次 `invoke()` 都是一次新的图运行，只是可以从同一 `thread_id` 的 checkpoint 恢复状态。

## 11. 先用三个小实验验证理解

运行 demo 前，需要先按项目配置好 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL`，然后执行：

```bash
.venv/bin/python -m src.agent.lc_mem_agent
```

模型的具体表述可能每次不同，重点观察状态和工具行为。

1. 把第三次调用改成另一个 `thread_id`，观察它是否还知道“张三”。
2. 删掉 `Annotated[..., add_messages]` 中的 reducer，对比每轮状态的变化。
3. 完成一次运行后，先注释掉“我叫张三”这次调用，再用相同 `thread_id` 重启 Python 程序并直接询问姓名，观察 `InMemorySaver` 的记忆是否还在。

这三个实验分别对应会话隔离、状态合并和存储生命周期，比直接背 API 更容易建立直觉。

## 12. 下一步学什么

看懂当前 demo 后，再按以下顺序逐步增加能力：

1. 增加第二个普通 Node，感受状态如何在节点间传递。
2. 学习条件边，让 State 决定执行哪条路径。
3. 把工具调用拆成显式的 Tool Node，看懂 Agent 的循环。
4. 学习持久化 checkpointer、状态查询和执行恢复。
5. 最后再看 interrupt、人工审批、子图和长期记忆。

到这里只需要能用一句话解释当前 demo：

> 外层 LangGraph 把对话消息放进 State，用一个 Node 调用已有 Agent，再通过 checkpointer 和 `thread_id` 在多次调用之间恢复同一份会话状态。
