# Agent 中间件：在模型调用前后插入规则

Tool 篇里的 Agent 会重复执行“模型判断 → 工具执行 → 模型继续回答”。中间件（Middleware）让你不用改 Agent 主循环，也能在关键位置做日志、脱敏、摘要、限流或重试。

这篇使用 LangChain 1.x 的 `create_agent`。不同 LangChain 版本的中间件参数会变化，运行前应以 `requirements.txt` 锁定的版本和本地安装结果为准。

## 1. 中间件放在哪里

最常用的三个时机：

| 位置 | 适合做什么 |
|---|---|
| `before_model` | 脱敏、裁剪历史、补充请求信息 |
| `after_model` | 记录结果、审核输出、打标 |
| `wrap_model_call` | 计时、缓存、重试、动态选择模型 |

`before_model` 和 `after_model` 是前后两个钩子；`wrap_model_call` 会拿到真正执行模型的 `handler`，因此能把调用包起来。

## 2. 从一个日志中间件开始

```python
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"即将调用模型：{len(state['messages'])} 条消息")
        return None

    def after_model(self, state, runtime):
        print(f"模型返回：{state['messages'][-1].content}")
        return None
```

注册时传入列表：

```python
agent = create_agent(model=model, middleware=[LoggingMiddleware()])
```

返回 `None` 表示不改状态，继续执行。多个中间件会按列表顺序组合；日志里不要直接打印用户消息、Token 或连接信息，尤其不能把隐私当成“方便调试”的代价。

## 3. 装饰器写法与环绕调用

无状态的简单钩子也可以写成函数：

```python
@before_model
def log_before_model(state, runtime):
    print(f"即将调用模型：{len(state['messages'])} 条消息")
    return None

@wrap_model_call
def measure(request, handler):
    started = time.perf_counter()
    result = handler(request)
    print(f"模型调用耗时：{time.perf_counter() - started:.2f}s")
    return result
```

`handler(request)` 是实际模型调用。省略这一行就等于拦截请求；调用两次就会发出两次模型请求。它适合计时、失败重试和缓存，但重试时要考虑工具调用是否已经产生副作用。

类写法适合需要保存配置或多个钩子的情况；装饰器写法适合一小段无状态逻辑。项目中的 [custom_middleware_by_class.py](custom_middleware_by_class.py) 与 [custom_middleware_by_decorators.py](custom_middleware_by_decorators.py) 展示了两种等价入口。

## 4. 脱敏：模型和日志都不该看到原文

[custom_middleware_demo.py](custom_middleware_demo.py) 的 `DesensitizeDataMiddleware` 在模型调用前把邮箱和中国大陆手机号替换为 `[EMAIL]`、`[PHONE]`：

```python
class DesensitizeDataMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        processed_count = 0
        for message in state.get("messages", []):
            if not isinstance(message.content, str):
                continue
            sanitized = self._desensitize_text(message.content)
            if sanitized != message.content:
                message.content = sanitized
                processed_count += 1

        if processed_count:
            print(f"已脱敏 {processed_count} 条消息")
        return state
```

这个示例刻意做到几件事：

- 检查每一条消息，而不是处理第一条就返回；
- 已经含有占位符时不再替换，重复经过中间件不会继续改写；
- 只记录处理数量，不记录原始邮箱、电话或完整用户输入；
- 正则只是一层基础保护，真实系统还要依据地区、字段来源和合规要求设计。

中间件并不能自动保证“所有数据都没有泄露”。在日志、工具参数、异常处理、追踪平台和存储层都要使用同样的最小暴露原则。

## 5. 长对话摘要

带历史的 Agent 会把旧消息不断带回模型。历史太长会增加 token、延迟和上下文溢出的风险。`SummarizationMiddleware` 的做法是在到达阈值时，把较早的消息总结成短摘要，只保留少数最近消息：

```python
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 80),
            keep=("messages", 1),
            summary_prompt="请简洁总结以下对话，保留关键信息：{messages}",
        )
    ],
)
```

`trigger` 是触发水位线，`keep` 是摘要时保留的最新完整消息数。示例把阈值设得很小，只是为了容易观察触发；真实阈值要留出模型输出和工具调用的空间。

当前依赖版本若仍只接受 `max_tokens_before_summary` 和 `messages_to_keep`，请使用该版本的参数名，或升级依赖后再用 `trigger`/`keep`。不能把“会显示弃用警告”当成通用保证，API 版本差异可能直接导致构造失败。

摘要会额外调用一次模型，也可能丢失细节。它适合长会话的上下文治理，不适合把精确交易记录、权限结论或原始证据只保存在摘要里。

## 6. 组合中间件时的顺序

一个常见顺序是：

```text
脱敏 → 请求日志/计时 → 模型调用 → 输出审核 → 结果日志
```

脱敏应在日志和模型调用之前，否则日志已经拿到敏感信息。摘要通常也应在进入模型前完成。每个中间件都应该只解决一个清楚的问题，避免在单个类里同时修改消息、调用外部服务、写数据库和重试。

## 7. 常见误解

- **中间件不是权限系统。** 它可以拒绝或改写请求，但数据库、文件和工具仍应有独立权限边界。
- **`thread_id` 不是用户认证。** 它只是把状态隔离的键；不能让客户端随意猜测别人的 id。
- **`after_model` 不等于安全审查完成。** 模型已经看过输入，也可能已经决定调用工具；高风险操作应在工具执行前再次确认。

到这里，Agent 的输入、模型调用和输出都有了可插入规则的位置。若 RAG 的问题出在“找错资料”，中间件救不了检索质量；下一篇会回到 RAG 基线，按失败模式选择更合适的索引、召回和后处理方式。
