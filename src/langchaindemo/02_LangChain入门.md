# LangChain 入门：把模型调用组装成 chain

前一篇手动写了 RAG 的每一步。这一篇先不急着做完整应用，看看 LangChain 怎样把“提示词 → 模型 → 输出处理”组合成可以重复调用的 `chain`。

项目使用 LangChain 1.x。示例模型由 [model.py](model.py) 中的 `getModel()` 创建，配置来自 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL`。这些变量指向 OpenAI-compatible 服务时，调用方式不变。

## 1. 一次模型调用

```python
model = getModel()
message = model.invoke("请用一句话介绍 LangChain")
print(message.content)
```

`invoke()` 返回的是消息对象，不是普通字符串；文本在 `.content` 中。模型生成的具体句子可能每次不同，不要把某次输出当成 API 契约。

## 2. Prompt：把变化的内容留给输入

最简单的 Prompt 是一个带变量的模板：

```python
prompt = PromptTemplate.from_template(
    "你是一个翻译助手，请将以下内容翻译成{language}：{text}"
)
message = model.invoke(prompt.format(language="中文", text="I am a programmer"))
```

模板把指令和输入分开了。以后要改角色或格式，通常只改模板。

对话模型还可以区分角色：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手，请将内容翻译成{language}"),
    ("human", "{text}"),
])
```

`system` 放稳定的行为要求，`human` 放本轮问题。元组写法和 `SystemMessagePromptTemplate` 的完整写法等价，初学时选一种即可。

### 需要例子时再用 Few-shot

如果规则很难用一句话描述，可以给模型几组输入和输出：

```python
prompt = FewShotPromptTemplate(
    examples=[
        {"input": "2+2", "output": "4", "description": "加法运算"},
        {"input": "5-2", "output": "3", "description": "减法运算"},
    ],
    example_prompt=PromptTemplate.from_template(
        "算式：{input} 值：{output} 类型：{description}"
    ),
    prefix="判断算式类型。",
    suffix="现在判断：{input}，值为{output}",
    input_variables=["input", "output"],
)
```

Few-shot 教的是模式，不是事实。示例里有占位符，模型也可能照抄；客服话术尤其要注意示范数据是否干净。

模板也可以先固定一部分变量：`prompt.partial(date="2026-08-11")` 会返回一个还需要其它变量的新模板。只有确实存在“启动时固定、请求时变化”的变量时才需要它。

## 3. 输出解析：让结果能被程序使用

模型输出通常是文本，但程序可能需要列表、JSON 或日期。parser 把模型消息转换成目标类型：

```python
chain = prompt | model | StrOutputParser()
print(chain.invoke({"language": "中文", "text": "I am a programmer"}))
```

常用 parser：

| parser | 得到 |
|---|---|
| `StrOutputParser` | 字符串 |
| `CommaSeparatedListOutputParser` | 字符串列表 |
| `JsonOutputParser` | Python 字典 |
| `DatetimeOutputParser` | `datetime` |

需要严格结构时，可以用 parser 提供的格式说明，或者使用 `with_structured_output` 配合 Pydantic 模型。两者都不能让模型变成绝对可靠；生产代码仍需捕获解析失败。

## 4. LCEL：用 `|` 连接 Runnable

能被 `.invoke()` 调用的组件都可以接进 LCEL。下面的 `|` 表示前一个组件的输出交给后一个组件：

```python
chain = prompt | model | StrOutputParser()
result = chain.invoke({"language": "中文", "text": "I am a programmer"})
```

这就是一个 `RunnableSequence`。写成显式形式也可以：

```python
chain = RunnableSequence(prompt, model, StrOutputParser())
```

`chain` 这个名字只是变量名，重点是每个阶段的输入输出要接得上。

### 一进一出：invoke、batch、stream

```python
chain.invoke(input_data)
chain.batch([input_a, input_b], config={"max_concurrency": 5})
for chunk in chain.stream(input_data):
    print(chunk, end="", flush=True)
```

`batch` 适合互不依赖的请求；并发数只是上限，实际耗时还受模型服务和限流影响。带会话状态的操作不要随意改成并发 batch。`stream` 是否逐 token 输出，取决于模型和 chain 中各组件的支持情况。

### 一个输入分成多路：RunnableParallel

```python
chain = RunnableParallel(
    length=lambda text: len(text),
    upper=lambda text: text.upper(),
)
print(chain.invoke("Hello"))
# {"length": 5, "upper": "HELLO"}
```

每个分支收到同一份输入，结果按名字组成字典。字典直接出现在 `|` 管道中时，也会被当作并行分支处理。

普通函数没有 `.invoke()`，需要用 `RunnableLambda` 适配：

```python
length_chain = RunnableLambda(len)
```

也可以用 `@chain` 装饰函数，让函数本身成为 Runnable。装饰器只是注册/包装动作，不需要在这里一次记住所有写法，看到示例时按输入输出理解即可。

### 保留原输入：RunnablePassthrough

```python
rag_input = RunnableParallel(
    question=RunnablePassthrough(),
    context=retrieve_context,
)
```

它让问题原样保留下来，同时生成检索上下文。`.assign()` 则是在字典上追加字段：

```python
chain = RunnablePassthrough().assign(
    length=lambda value: len(value["text"])
)
```

这是 RAG chain 常见的形状：问题一路去 retriever，另一路留给 Prompt，最后两个结果汇合。

## 5. 把 Document 塞进 Prompt

`create_stuff_documents_chain` 的职责很单一：把 Document 列表的正文拼起来，填入 `{context}`。它适合文档数量和总长度可控的场景。

```python
chain = create_stuff_documents_chain(model, prompt)
chain.invoke({"input": "文章讲了什么？", "context": documents})
```

如果需要控制格式，可以先显式提取 `doc.page_content` 再拼接，不要依赖复杂对象的隐式字符串表示。

## 6. 把 chain 变成服务

LangServe 可以把现有 chain 挂到 FastAPI：

```python
add_routes(app, chain, path="/langchainServer")
```

客户端请求通常是：

```json
{"input": {"language": "英文", "text": "我喜欢编程"}}
```

这部分需要单独启动服务，且 playground 不应直接暴露到生产环境。`RemoteRunnable` 可以让客户端用类似本地 chain 的方式调用远程服务。

## 7. 记忆：历史必须由外部带回

模型本身不会因为上一次 `invoke()` 就记住内容。最直观的做法是把历史消息再次放进 Prompt：

```python
history.add_user_message(user_input)
response = chain.invoke({"messages": history.messages})
history.add_ai_message(response)
```

`RunnableWithMessageHistory` 可以自动完成读取和追加。多用户场景必须根据 `session_id` 返回不同的历史存储；固定返回同一份 history 只能用于单会话演示。Redis 版本需要本地 Redis 服务和额外配置，不能把已有 Redis 数据当成示例必然结果。

## 8. 综合应用：放到最后看

客服示例把几条独立分析放进 `RunnableParallel`，再用 `.assign()` 汇合；订单号这种固定格式优先用正则，LLM 只负责正则无法处理的部分；JSON 解析失败时要有可接受的降级结果。

这些是把组件放进真实业务的方式，不是学习 LCEL 的必经 API。先能看懂上面的“输入 → 分支 → 汇合 → 输出”，再打开 `demo/customer/` 会轻松很多。

## 9. 下一篇

下一篇把这套组件接到真实的 Word 和网页知识源。你会看到手写 RAG 中的“文本块”在 LangChain 里变成 `Document`，而 retriever 可以直接作为 Runnable 接进 chain。
