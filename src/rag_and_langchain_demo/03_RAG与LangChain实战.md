# RAG 与 LangChain 实战：接入真实文档

前一篇讲了 Prompt、parser 和 LCEL。这一篇把它们接到真实知识源：Word 文档和网页。RAG 的骨架没有变，只是 Loader 和 retriever 替我们处理了更多细节。

## 1. 先认识 Document

LangChain 用 `Document` 表示一段知识：

```python
Document(page_content="正文", metadata={"source": "manual.docx"})
```

不同 Loader 负责把 Word、网页、PDF 等格式转换成 Document。后续切分、embedding 和检索不需要知道原文件是什么格式。

## 2. 本地 Word：从加载到检索

[doc_and_llm.py](doc_and_llm.py) 使用 `Docx2txtLoader`。脚本通过 `Path(__file__)` 定位同目录文档，因此从仓库根目录或其它当前目录运行都不会因为 CWD 改变而找不到文件。

```python
loader = Docx2txtLoader(str(document_path))
documents = loader.load()
split_documents = splitter.split_documents(documents)
vector_store = Chroma.from_documents(split_documents, embedding)
```

`split_documents()` 会把原文切成多个 Document，并继承 metadata。`Chroma.from_documents()` 将 embedding 和写入向量库合在了一步；想观察手写版本的每个阶段，可以回看 01 篇。

### retriever 是一个 Runnable

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)
```

检索器接收问题，返回 Document 列表。它可以直接放进 LCEL 的并行分支：

```python
chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(
            lambda docs: "\n\n".join(doc.page_content for doc in docs)
        ),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

这里显式提取 `page_content`，所以 Prompt 看到的是正文而不是 Document 的调试表示。

## 3. 三种检索参数

```python
vector_store.as_retriever()
```

默认是普通相似度检索，适合先跑通。

```python
vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.4},
)
```

阈值检索会丢弃低于阈值的结果，但阈值和 embedding 模型、语料分布有关，需要用自己的问题集调试。

```python
vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)
```

MMR 先取 `fetch_k` 个候选，再兼顾问题相关性和结果之间的差异，避免几段几乎重复的内容占满上下文。`lambda_mult` 越接近 1 越偏相关性，越接近 0 越偏多样性。它不是“必然更好”的开关，应结合数据比较。

## 4. 网页：先清掉导航噪声

[rag_with_webpage.py](rag_with_webpage.py) 使用 `WebBaseLoader`，并通过 `SoupStrainer` 只解析网页正文：

```python
loader = WebBaseLoader(
    web_path="https://www.gov.cn/lianbo/202608/content_7078029.htm",
    bs_kwargs={"parse_only": bs4.SoupStrainer(id="UCAP-CONTENT")},
)
```

网页导航、页脚和推荐内容会稀释 embedding；正文过滤往往比调整 top-k 更值得先做。网页内容会变化，运行结果、块数和答案不能当作永久不变的数字。Loader 产生的 source metadata 也应保留下来，方便回答时展示来源。

## 5. 两种 RAG chain 组装方式

### 手拼 LCEL

```python
chain = {
    "question": RunnablePassthrough(),
    "context": retriever | format_documents,
} | prompt | model | StrOutputParser()
```

每一步都暴露出来，适合学习和定制。

### 使用标准组件

```python
stuff_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, stuff_chain)
response = rag_chain.invoke({"input": "发布会讲了什么？"})
```

标准组件约定 Prompt 中有 `{context}` 和 `{input}`。外层通常返回：

```python
{
    "input": "...",
    "context": [Document(...)],
    "answer": "...",
}
```

这样前端可以同时拿到原问题、证据和答案。当前项目使用 `langchain_classic` 中的这两个组件；它们用于兼容经典 chain 写法。新代码应查看当前 LangChain 版本对应的导入位置，不要只复制旧教程的 import。

## 6. 文档、证据和答案

Prompt 中要明确限制模型：只依据上下文回答，找不到依据就说不知道。检索结果的 metadata 可以用于显示来源，但不要因为某个 source 字段存在，就假定答案一定正确；仍要检查正文是否支持答案。

一个实用的调试顺序是：

1. 直接打印 retriever 返回的 Document；
2. 检查问题真正需要的信息是否在这些正文中；
3. 再检查 Prompt 和模型回答。

如果第 1 步就错了，改模型提示词没有用。

## 7. 从实验代码走向应用

示例每次启动都重新加载文档、切分并建临时向量库，适合学习。真实系统一般把索引任务离线化：文档变化时增量更新，提问请求只做检索和生成。Word 需要 `docx2txt`，网页需要 `beautifulsoup4` 等依赖；外部网页、模型和 embedding 服务不可用时，示例无法完成端到端运行，这属于运行条件而不是代码输出。

## 8. 下一篇

工具调用解决的是“模型需要外部能力”时怎么办；模型只生成工具请求，是否执行以及怎样执行，仍由程序控制。接下来进入 Tool 与 Function Calling。
