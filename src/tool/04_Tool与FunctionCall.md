# Tool 与 Function Calling：模型提出请求，程序决定执行

模型能写代码，不代表它能直接访问你的电脑、数据库或实时网络。Tool calling 解决的是一个很具体的问题：让模型知道有哪些外部能力，同时把真正的执行权留在程序里。

> 模型只负责提出“请调用哪个工具、参数是什么”；你的程序负责校验、执行，并把结果作为消息交还给模型。

## 1. 没有工具时，模型拿不到实时信息

直接问模型“今天是几月几号”，它可能猜一个日期，也可能说自己无法获取当前日期。无论哪种情况，都不能当作时钟使用。实时日期应该由 Python 提供。

## 2. 定义一个工具

```python
import datetime
from langchain_core.tools import tool

@tool
def get_date() -> str:
    """获取今天的具体日期。"""
    return datetime.date.today().strftime("%Y-%m-%d")
```

`@tool` 把普通函数注册成 Tool。函数名、参数类型和 docstring 会组成给模型看的说明书：

- 名字告诉模型工具叫什么；
- 类型注解帮助它生成参数；
- docstring 说明什么时候使用，以及返回什么。

说明书必须和真实行为一致。工具叫“查天气”，却返回固定的推荐文案，模型无法替你发现这个错误。

## 3. `bind_tools()` 只把菜单交给模型

```python
tool_model = model.bind_tools([get_date])
response = tool_model.invoke("今天是几月几号？")

print(response.content)
print(response.tool_calls)
```

这一轮通常会看到：

```text
content: ""
tool_calls: [{"name": "get_date", "args": {}, "id": "call_..."}]
```

注意，`get_date()` 还没有被调用。模型只返回了一个请求：工具名、参数和本次调用的 id。程序如果自己使用 `bind_tools()`，就需要读取 `tool_calls`、执行对应函数、构造 ToolMessage，再把消息发回模型。

## 4. `create_agent()` 自动完成调用循环

手动处理消息很容易漏掉工具调用 id 或消息顺序。Agent 把这个循环包装起来：

```python
from langchain.agents import create_agent

agent = create_agent(model=model, tools=[get_date])
result = agent.invoke({
    "messages": [{"role": "user", "content": "今天是几月几号？"}]
})
print(result["messages"][-1].content)
```

一次调用内部大致经过四拍：

```text
HumanMessage：用户的问题
AIMessage：请求调用 get_date
ToolMessage：get_date 的返回值
AIMessage：模型根据返回值生成最终回答
```

`ToolMessage.tool_call_id` 用来对应具体请求。工具需要参数时，模型会按照 schema 填入参数，例如 `open_browser(url="https://...")`；这不意味着参数可以直接信任，程序仍要在边界上校验。

## 5. 工具返回值是模型继续回答的依据

不要让有用的工具隐式返回 `None`：

```python
@tool
def open_browser(url: str) -> str:
    """打开浏览器访问指定网址；这是一个有副作用的操作。"""
    opened = webbrowser.open(url)
    return f"浏览器打开请求已发送：{url}（成功={opened}）"
```

返回 `None` 时，Agent 里可能出现 `ToolMessage(content='null')`。模型有时会据此自行说“已经完成”，但程序没有告诉它实际结果。明确返回成功或失败状态更容易调试。

浏览器、计算器、发消息、改数据等工具都会产生副作用。真实应用不要因为模型发出了 tool call 就自动执行，可以先向用户展示目标和参数，确认后再执行；网址还应限制协议、域名或允许列表。示例中的 `webbrowser.open()` 会打开本机浏览器，运行前要知道这一点。

## 6. 多个工具：模型依据说明书选择

```python
agent = create_agent(
    model=model,
    tools=[get_current_time, recom_drink, open_calc, open_browser],
)
```

用户说“现在几点了”时，模型可能选择时间工具；说“我渴了”时，是否选择饮品推荐工具取决于工具描述，而不是用户必须写出工具名。`function_call_demo_01.py` 用 Gradio 展示了这一点，也包含按操作系统打开计算器的示例。它是交互演示，不是安全的生产工具授权方案。

## 7. 有记忆的 Agent

短期记忆可以使用 LangGraph 的 checkpointer：

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[get_date],
    checkpointer=InMemorySaver(),
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "记住我喜欢篮球"}]},
    config={"configurable": {"thread_id": "user_1"}},
)
```

同一个 `thread_id` 的多轮调用会带上之前的消息，包括 ToolMessage。不同用户必须使用不同 id；否则历史会串在一起。`InMemorySaver` 只适合当前进程，跨进程需要 Redis 或其它持久化存储，并且要设置过期和权限策略。

## 8. 两种工具定义方式

现代代码优先使用 `@tool`，因为参数 schema 可以从类型注解生成：

```python
@tool
def search_order(order_id: str) -> str:
    """根据订单号查询订单状态。"""
    ...
```

旧式 `Tool(name, func, description)` 仍常见于兼容代码：

```python
Tool(
    name="get_current_time",
    func=get_current_time,
    description="当你想知道现在的时间时可以使用",
)
```

这里的 `Tool` 更适合一个简单输入；需要多个结构化参数时，`@tool` 的 schema 更清楚。无论写法怎样，description/docstring 都不能替代程序侧的权限和参数校验。

## 9. `load_tools()`：使用现成工具

社区包提供了一些现成工具：

```python
from langchain_community.agent_toolkits.load_tools import load_tools

tools = load_tools(["arxiv"])
agent = create_agent(model=model, tools=tools)
```

`function_call_demo_02.py` 用 arXiv 论文编号查询做示例。它需要访问外网，遇到限流或网络失败时，失败发生在工具执行阶段，不是模型突然失去了工具能力。

## 10. Text-to-SQL：模型生成指令，程序执行指令

数据库场景可以复用工具的分工，但不要把数据库权限直接交给模型：

```text
自然语言问题
    ↓
模型生成 SQL（只生成，不执行）
    ↓
程序检查 SQL 是否为允许的只读语句
    ↓
数据库用只读账户执行
    ↓
查询结果回到模型，生成自然语言回答
```

`create_sql_query_chain` 会把表结构和样例行提供给模型，它负责生成 SQL，不负责安全授权。项目示例 `use_tools_query_sql.py` 还在执行前调用 `validate_read_only_sql()`，拒绝多语句、写入语句和管理语句。

这三层防线缺一不可：

1. SQL 文本校验，防止模型生成明显危险的语句；
2. 数据库账户只授予 `SELECT` 权限，防止校验遗漏造成写入；
3. 限制可访问的表、字段和返回行数，避免越权读取。

不能说 `QuerySQLDataBaseTool` 默认只执行 SELECT；是否安全取决于执行器、数据库权限和外围校验。不要把用户输入直接拼进 SQL，也不要把数据库错误原文和连接配置返回给用户。

## 11. 调试 Agent 的方法

遇到“模型没有调用工具”时，按顺序检查：

- 工具名和说明是否准确；
- 参数是否有类型注解和必填约束；
- 模型响应的 `tool_calls` 是否为空；
- 工具是否真的执行、返回值是否有效；
- ToolMessage 是否带上对应的调用 id；
- 失败是不是来自外部网络、浏览器或数据库。

下一篇会处理另一个问题：工具调用循环跑起来后，怎样在模型调用前后统一做日志、脱敏、摘要和拦截。
