# 基础 RAG：从文档到答案

这篇先不使用 LangChain，手动走一遍 RAG。这样做不是为了以后拒绝框架，而是为了知道框架替我们省掉了哪些步骤。

## 1. RAG 到底做了什么

RAG（Retrieval-Augmented Generation，检索增强生成）可以拆成两段：

- **索引**：读取文档，切成小块，为每个块生成向量并保存。
- **问答**：把问题也变成向量，找出最相关的几个块，再把问题和这些块交给模型。

模型不需要把整篇文档都记住。它只在回答这一问时看到检索出来的资料。

本篇对应的代码如下：

| 文件 | 作用 |
|---|---|
| [text_splitter.py](text_splitter.py) | 递归切分文本 |
| [embedding.py](embedding.py) | 调用本地或在线 embedding 服务 |
| [vector_store.py](vector_store.py) | 写入和查询 Chroma |
| [llm.py](llm.py) | 调用对话模型 |
| [main.py](main.py) | 编排完整流程 |

从仓库根目录运行：

```bash
python -m src.rag.main
```

这个命令需要 `.env` 中有对话模型和 embedding 服务配置。默认的 `main.py` 使用在线 embedding；想用 Ollama 时，在 `create_embedder()` 中切换到 `LocalEmbedder`，并使用 Ollama 原生地址 `http://127.0.0.1:11434`。OpenAI-compatible 服务通常使用带 `/v1` 的地址，两者不要混用。

## 2. 先看完整流程

`main.py` 的 `run_rag_demo()` 把一次问答分成几步：

```text
rag_sample.txt
    │
    ▼
读取文本 → 分块 → 为每个块生成向量 → 写入 Chroma
                                      │
用户问题 → 生成问题向量 → 查询最相近的块 ─┘
                                      │
                                      ▼
                         资料 + 问题 → 对话模型
```

索引部分在用户提问前完成；示例使用内存 Chroma，所以程序退出后索引会消失。真实服务通常会换成持久化存储，并且只在文档发生变化时重新索引。

## 3. 分块：不要把整篇文档当成一条记录

向量模型有输入长度限制。更重要的是，整篇文档通常包含很多主题，算出来的向量会混合这些主题，检索不容易定位。

`RecursiveTextSplitter` 按这个顺序寻找切分点：段落、换行、中文句号和问号、分号、逗号、空格，最后才逐字符切分。`chunk_size=512` 表示最多 512 个 Python 字符，`chunk_overlap=64` 表示相邻块保留一小段重叠内容。

```python
splitter = RecursiveTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split(text)
```

重叠的用途很实际：一句话刚好跨过边界时，下一块仍能看到前面的部分。`chunk_overlap` 必须小于 `chunk_size`，这个约束在构造对象时会检查。

可以先不记住所有分隔符。只要知道：它会尽量保留段落和句子边界，实在太长才继续切细。

## 4. 向量：把“像不像”变成距离

embedding 模型把一段文本转换成固定长度的数字列表，例如 1024 个数字。意思相近的文本，通常会在这个向量空间里更接近。

文档和问题必须使用同一个 embedding 模型。文档用模型 A、问题用模型 B，就像拿两张不同坐标系的地图比较位置，距离没有意义。

项目用 `Embedder` Protocol 约定统一接口：

```python
vectors = embedder.embed(chunks)
question_vector = embedder.embed([question])[0]
```

接口接收一批字符串并保持顺序。实现会检查输入类型和返回数量，避免服务少返回一个向量后，文本和向量错位入库。

## 5. 向量库：保存并找最近邻

Chroma 负责保存文本和向量，并按照余弦距离查询近邻。余弦距离关注两个向量的方向是否接近，适合比较文本语义。

本项目显式把向量传给 Chroma：

```python
store = ChromaVectorStore.in_memory()
store.add_documents(chunks, chunk_embeddings)
retrieved = store.query(question_embedding, n_results=3)
```

这样可以清楚看到 embedding 发生在哪里，也能保证写入和查询使用同一个 embedding 实现。Chroma 内部可以使用 HNSW 等近似最近邻索引，但具体速度和召回率取决于数据量和参数，不要把它理解成“永远只需要对数次比较”。

`query()` 返回的是一个字符串列表；Chroma 底层因为支持批量查询，原始返回结构会多一层列表，封装类替我们取出当前这一个问题的结果。

## 6. 给模型的 Prompt

召回资料后，不要把问题单独交给模型，而是明确告诉模型只能依据资料回答：

```python
def build_rag_prompt(question, contexts):
    context_text = "\n\n".join(
        f"[片段 {index}]\n{context}"
        for index, context in enumerate(contexts, start=1)
    )
    return (
        "请仅根据下面的参考资料回答问题。"
        "如果资料中没有答案，请明确回答‘根据现有资料无法确定’。\n\n"
        f"参考资料：\n{context_text}\n\n"
        f"问题：{question}"
    )
```

片段编号方便调试和后续引用。拒答要求也很重要：检索没有找到依据时，模型应该说不知道，而不是补一个听起来合理的答案。

## 7. 一次运行看什么

`main.py` 会打印分块数量、向量维度、召回片段和模型回答。分块数量和向量维度在同一份数据、同一配置下比较稳定；回答措辞、召回排序和耗时会受模型版本、网络和服务端参数影响，不能当成固定输出。

问题“番茄区在什么条件下会开始灌溉？”的示例回答会涉及土壤含水率下限和高蒸发时段。这些内容应该能在打印出的召回片段中找到；如果找不到，就该检查分块、embedding 和 top-k，而不是先修改模型 Prompt。

## 8. top-k 怎么理解

`top_k=3` 是一个取舍：

- 太小，资料可能不完整；
- 太大，噪声、上下文长度和调用成本会上升。

`top-k=1` 的对照实验很适合发现这个问题，但它不是普遍更准确的配置。后面会用 MMR、混合检索和重排序，在“多召回”和“少噪声”之间做得更细。

## 9. 继续学习

这一篇解决“RAG 每一步发生什么”。接下来按这个顺序学习：

1. [LangChain 入门](../langchaindemo/02_LangChain入门.md)：用 Prompt、模型和 parser 组装 `chain`。
2. [RAG 与 LangChain 实战](../rag_and_langchain_demo/03_RAG与LangChain实战.md)：让 Loader、Document 和 retriever 接管重复代码。
3. [Tool 与 Function Calling](../tool/04_Tool与FunctionCall.md)：让模型提出工具调用请求，由程序执行。
4. [Agent 中间件](../middleware/05_Agent中间件.md)：在模型调用前后加规则。
5. [Advanced RAG](../advance_rag/06_Advanced%20RAG.md)：针对基线的具体失手位置做检索优化。
