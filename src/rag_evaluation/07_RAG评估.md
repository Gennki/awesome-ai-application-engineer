# 07 RAG 评估：从指标到优化闭环

RAG（检索增强生成）系统由两部分协作完成：检索器从知识库找出上下文，生成器据此回答问题。只看最终回答“感觉不错”并不能说明系统可靠：回答错误时，我们还需要知道问题出在知识库、检索、提示词，还是模型生成。

本章使用 [Ragas](https://docs.ragas.io/) 对仓库中的 RAG 示例进行评估，并通过同一份汽车手册数据比较分块、混合检索、上下文压缩和重排序。示例代码位于 `src/rag_evaluation/`。

> Ragas 会让评测模型和 embedding 模型参与计算，因此一次评测通常比一次普通问答产生更多模型请求。请先用少量、有人工标注的问题验证流程，再逐步扩大测试集。

## 学习目标

完成本章后，你将能够：

1. 区分检索质量和生成质量，理解四个常用 RAG 指标；
2. 准备包含问题、回答、检索上下文和参考答案的评测集；
3. 使用 Ragas 评估一个已有的 RAG 应用；
4. 根据分数定位检索器或生成器的瓶颈，而不是盲目调整模型；
5. 用同一份评测集比较不同的 `chunk_size`、检索器和重排序策略。

---

## 1. 为什么要评估 RAG

传统程序的许多输出是确定的，例如金额、状态码或排序结果；RAG 的输入和输出都是自然语言，同一问题在不同模型版本、提示词或检索结果下可能得到不同回答。因此评估至少解决以下问题：

- **上线前验收**：回答是否正确、基于证据且切题？
- **优化决策**：修改分块、Top-K 或提示词后，效果改善了多少？
- **知识库维护**：新增或删除文档后，是否引入噪声或遗漏关键信息？
- **模型变更评估**：更换对话模型、embedding 模型或评测模型后，质量是否变化？

人工评估仍然不可替代，特别是在医疗、法律等高风险场景。自动评估的价值在于：让同一批样本能够被快速、重复地运行，作为实验筛选和持续回归测试的基础。合理的流程是“自动评分发现候选问题 + 人工抽样确认原因”，而不是把单个自动分数当作绝对事实。

---

## 2. 评估什么：检索与生成两个层次

一次 RAG 问答可以抽象为：

```text
用户问题 question
       │
       ▼
检索器 ──────────────► 上下文 contexts
       │                        │
       └────────────────────────┤
                                ▼
                         生成器 ─────► 回答 answer

人工标注的正确答案 ground_truth ─────► 评测器
```

因此，评测应分为两类：

- **检索评估**：上下文是否找全、找准，并且重要内容是否排在前面；
- **响应评估**：回答是否能由上下文支持，是否直接回答了问题。

本仓库的示例使用以下四个指标：

| 指标 | 层次 | 主要输入 | 它回答的问题 |
| --- | --- | --- | --- |
| `context_recall` | 检索 | `contexts`、`ground_truth` | 回答问题所需的信息是否被找全？ |
| `context_precision` | 检索 | `contexts`、`ground_truth` | 找回的内容有多少有用，且是否排在前面？ |
| `faithfulness` | 生成 | `answer`、`contexts` | 回答是否有上下文依据，是否产生幻觉？ |
| `answer_relevancy` | 生成 | `question`、`answer` | 回答是否直接、完整地回应用户问题？ |

### 2.1 上下文召回率：`context_recall`

上下文召回率衡量检索结果是否覆盖参考答案中的关键事实。问题“《三体》的作者是谁、何时出版？”，参考答案包含“刘慈欣”和“2006 年”两个事实：

- 上下文同时包含两项信息，召回率较高；
- 上下文只出现“刘慈欣”，则缺少出版时间，召回率会下降。

低召回通常意味着知识库缺文档、分块切断了关键信息、查询表达与文档不匹配，或 Top-K 过小。

### 2.2 上下文精度：`context_precision`

上下文精度关注返回内容的信噪比和排序。对于“苹果公司总部在哪里？”，包含“总部位于加利福尼亚州库比蒂诺”的片段有帮助，成立年份、CEO、产品列表则是噪声。

Ragas 的上下文精度不仅在意相关内容比例，也考虑排序：相同数量的相关片段，排在第 1、3 位通常优于全部被无关片段挤到末尾。低精度常见于分块过大、Top-K 过高、召回策略引入过多噪声，或相关文档没有被重排序到前列。

### 2.3 忠实度：`faithfulness`

忠实度检查回答中的原子事实能否从检索上下文推导出来。它评估的是“是否基于证据回答”，不等价于“事实在现实世界是否正确”。

例如上下文写着“爱因斯坦于 1921 年获得诺贝尔物理学奖，获奖理由与光电效应有关”，回答“他于 1921 年因光电效应获奖”具有依据；若额外补充上下文中不存在的任职经历，即使该信息真实，也会降低忠实度。

### 2.4 答案相关性：`answer_relevancy`

答案相关性检查回答是否正面回应用户的问题，不直接判断事实正确性。例如用户问“如何煮咖啡”，回答研磨、温度和冲泡时间，相关性较高；回答咖啡的起源历史则答非所问。

### 2.5 用 F1 综合看检索质量

召回率和精度往往需要权衡：增加召回数量可能带来噪声，提高阈值又可能漏掉证据。仓库中的端到端示例会对每个问题的 `context_precision` 和 `context_recall` 求均值，再计算：

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

F1 有助于比较检索方案，但不能代替四项原始指标。例如业务优先级是“宁可多返回资料也不能漏条款”时，高召回可能比更高的 F1 更重要。

---

## 3. 评测数据集

当前示例使用 Ragas 旧版接口常见的四列：

```python
from datasets import Dataset

records = {
    "question": ["板蓝根颗粒的储存条件是什么？"],
    "answer": ["应密封保存，并放在阴凉干燥处。"],
    "contexts": [["板蓝根颗粒：【贮藏】密封。"]],
    "ground_truth": ["板蓝根颗粒需要在密封条件下储存。"],
}
dataset = Dataset.from_dict(records)
```

各字段的含义如下：

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `question` | 用户提问 | 真实日志、人工设计或自动生成 |
| `answer` | 待评估 RAG 的最终回答 | 待测系统生成 |
| `contexts` | 生成回答时实际传给模型的文本块列表 | 检索链输出；每条样本是一个列表 |
| `ground_truth` | 人工确认的参考答案 | 标注集或权威资料 |

`contexts` 必须记录**本次回答实际使用的上下文**，而不是事后挑选的理想文档；否则忠实度和检索指标无法反映真实线上路径。

> Ragas 不同版本的数据模式和指标 API 会变化。本项目脚本的 `question`、`answer`、`contexts`、`ground_truth` 字段应配合当前安装的版本使用。生产项目应在依赖文件中锁定 Ragas 及相关包版本；升级时先在固定评测集上验证字段映射和分数趋势。

---

## 4. 运行前准备

### 4.1 安装依赖

先在项目根目录安装基础依赖：

```bash
pip install -r requirements.txt
```

RAG 评估样例还直接依赖 Ragas、Datasets、PDF loader、FAISS 和 BM25。当前 `requirements.txt` 没有锁定这组包；首次运行可安装：

```bash
pip install ragas datasets pypdf faiss-cpu rank-bm25
```

建议在你自己的项目中把验证过的版本写入依赖清单，避免 Ragas 或 LangChain 升级导致 API 不兼容。

### 4.2 配置模型服务

将 `.env.example` 复制为 `.env`，再填入可用凭据：

```bash
cp .env.example .env
```

评估样例由 [`src/langchaindemo/model.py`](../langchaindemo/model.py) 读取下列配置：

| 配置 | 用途 |
| --- | --- |
| `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` | OpenAI 兼容的对话模型；既用于回答，也用于部分 Ragas 指标 |
| `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL` | embedding 模型；用于向量检索与需要 embedding 的指标 |
| `RERANK_API_KEY`、`RERANK_BASE_URL`、`RERANK_MODEL` | 仅重排序实验需要；接口采用 Cohere 兼容的 `/rerank` 格式 |

`automobile_handbook.py` 还支持以下可选变量：

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `RAG_TOP_K` | `4` | 压缩检索前由基础检索器返回的文档数量 |
| `OPENAI_TIMEOUT` | `180` | 回答模型请求超时（秒） |
| `OPENAI_MAX_RETRIES` | `2` | 回答模型最大重试次数 |
| `RAGAS_TIMEOUT` | `300` | 单次 Ragas 评测请求超时（秒） |
| `RAGAS_MAX_RETRIES` | `3` | Ragas 最大重试次数 |
| `RAGAS_MAX_WORKERS` | `1` | Ragas 评测并发数；默认串行，降低兼容接口超时概率 |

### 4.3 数据与索引

端到端实验读取：

- PDF：`src/rag_evaluation/data/初赛训练数据集.pdf`
- FAISS 索引：`src/rag_evaluation/data/faiss_index/`

首次运行会从 PDF 分块、调用 embedding 服务并创建索引；后续如果对应 `.faiss` 索引已存在，脚本会直接加载它。示例为了加载 FAISS 的本地持久化文件启用了 `allow_dangerous_deserialization=True`，因此只能加载自己创建或可信来源的索引文件，不能加载未知来源的序列化文件。

---

## 5. 实战 1：对静态样本做 Ragas 评测

文件：[`medical_evaluation.py`](medical_evaluation.py)

这个实验不执行检索和生成，而是预置了 10 条药品问答样本。每条样本包含问题、参考答案、检索上下文和待评估回答，适合理解四项指标分别在衡量什么。

从项目根目录运行：

```bash
python -m src.rag_evaluation.medical_evaluation
```

脚本的核心流程是：

1. 用 `Dataset.from_dict(data)` 将四列数据转换为 Hugging Face Dataset；
2. 通过 `getModel()`、`getEmbedding()` 获取本项目配置的模型；
3. 用 `LangchainLLMWrapper` 和 `LangchainEmbeddingsWrapper` 适配给 Ragas；
4. 调用 `evaluate()` 计算四项指标；
5. 输出 DataFrame，并在**运行命令所在目录**写入 `ragas_reval.csv`。

```python
result = evaluate(
    dataset=dataset,
    llm=vllm,
    embeddings=vallm_e,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)
```

可以重点观察样本中“回答比上下文多说了什么”。例如只提供“密封”这一上下文，却在回答中增加“阴凉干燥处”，该附加事实会让忠实度受到影响；它也说明“听起来合理”不是 RAG 回答的充分条件。

---

## 6. 实战 2：比较知识块大小

文件：[`chunk_size_evaluation.py`](chunk_size_evaluation.py)

分块大小决定一个检索单元携带多少上下文：

- 过小：术语、条件和前后逻辑可能被切开，造成低召回；
- 过大：一个块混入太多主题，语义向量不够聚焦，造成低精度；
- 合适的值依赖问题类型、文本结构、embedding 模型和 Top-K，不能从别的项目直接照搬。

该脚本执行的路径为：

```text
汽车手册 PDF
  → RecursiveCharacterTextSplitter（overlap = chunk_size × 20%）
  → FAISS 向量检索（top_k = 10）
  → 检索 chain 生成 answer 与实际 contexts
  → Ragas 四项评分 + 检索 F1
```

当前源码默认 `chunk_size = 512`，并且只运行纯 FAISS 检索；要比较 `128`、`256` 和 `512`，应逐次改动第 55 行附近的 `chunk_size`，保持问题集、参考答案、模型、embedding、分割器、Top-K 与其他配置不变。

此脚本的 PDF 和索引路径相对于 `src/rag_evaluation`，因此可这样运行：

```bash
cd src/rag_evaluation && PYTHONPATH=../.. python chunk_size_evaluation.py
```

每组实验至少记录：分块数、`chunk_size`、`chunk_overlap`、Top-K、模型与 embedding 版本、四项指标均值、F1、耗时和失败请求数。对于分数接近的方案，还应人工查看每个问题的检索片段与答案，不应只按小数点后的差异选择方案。

---

## 7. 实战 3：混合检索与上下文压缩

文件：[`automobile_handbook.py`](automobile_handbook.py)

该脚本固定以 128 字符为分块大小，20% overlap 处理汽车手册，并构造三类检索器：

1. **FAISS 检索器**：按 embedding 相似度召回；
2. **BM25 检索器**：按词项匹配召回；
3. **混合检索器**：`EnsembleRetriever` 按 `BM25:FAISS = 0.2:0.8` 集成二者。

混合检索将语义匹配和关键词匹配结合起来，常用于术语、型号、条款编号等关键词很重要的知识库；但它不必然提高每个指标，权重、Top-K 和数据质量都需要通过评测集验证。

在混合检索器之上，脚本再构建：

```python
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=mix_retriever,
)
```

`LLMChainExtractor` 会从初始召回文档中抽取与当前问题有关的内容，或删除无关文档。因此它可能降低上下文噪声、提高精度，但也可能压缩掉回答所需细节、增加一次模型调用成本。

从项目根目录运行：

```bash
python -m src.rag_evaluation.automobile_handbook
```

当前默认实际执行的是 `compression_retriever`；纯 FAISS 和未压缩混合检索的 `exec_eval(...)` 调用在源码中被注释。若要做公平对照，请一次只取消一个调用，并避免在同一轮中混合不同的模型或索引版本。

该脚本会打印每个问题的模型回答、传给生成器的上下文、四项 Ragas 指标和检索 F1。为兼容本地或代理服务，评测默认使用 `batch_size=1` 且 `RAGAS_MAX_WORKERS=1`；若服务稳定且额度允许，可审慎增加并发。

---

## 8. 实战 4：混合检索加重排序

文件：[`automobile_handbook_rebank.py`](automobile_handbook_rebank.py)

仅靠向量相似度或 BM25 的初步排序并不总能把最有用的片段放到前面。该脚本先用 128 字符块构建 FAISS 与 BM25 混合检索：

```python
mix_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weight=[0.2, 0.8],
)
```

随后使用 [`getReranker()`](../langchaindemo/model.py) 创建重排序器，并作为 `ContextualCompressionRetriever` 的 `base_compressor`：

```python
rerank_retriever = getReranker()  # 默认 top_n=5
combin_retriever = ContextualCompressionRetriever(
    base_compressor=rerank_retriever,
    base_retriever=mix_retriever,
)
```

流程是“混合召回 10 条 → 重排序接口返回最相关的 5 条 → 生成答案 → Ragas 评估”。这与上一节的 LLM 文本抽取不同：重排序保留的是完整的高相关文档，并将相关度分数写入文档元数据。

运行前需配置有效的 `RERANK_API_KEY`、`RERANK_BASE_URL` 和 `RERANK_MODEL`。脚本使用相对数据路径，运行命令为：

```bash
cd src/rag_evaluation && PYTHONPATH=../.. python automobile_handbook_rebank.py
```

重排序常用于提升 `context_precision`，但它可能把排序靠后的补充证据过滤掉，使 `context_recall` 下降。应同时检查四项指标与逐题样本，避免因只追求精度而漏掉复杂问题所需的多个事实。

---

## 9. 如何从分数定位问题

下表是排查的起点，而不是自动化结论。首先检查低分样本的原始问题、实际上下文和模型回答，再决定是否调整系统。

| 现象 | 优先检查 | 常见改进方向 |
| --- | --- | --- |
| `context_recall` 低 | 知识库是否覆盖；关键信息是否被切断；Top-K 是否过小 | 补充/清洗知识库，调整分块，查询改写，混合检索，提高 Top-K |
| `context_precision` 低 | 返回结果是否包含大量噪声；相关文档是否排在后面 | 调整 Top-K、相似度阈值、分块策略、混合权重，加入重排序或上下文压缩 |
| `faithfulness` 低 | 回答是否补充了上下文没有的事实；提示词是否允许自由发挥 | 要求仅根据上下文回答，不确定时明确拒答；改善上下文质量；增加生成后事实核查 |
| `answer_relevancy` 低 | 问题意图是否被理解；回答是否冗余或遗漏子问题 | 问题分类/改写，结构化提示，先分解复杂问题，压缩冗余上下文 |
| 四项均低 | 评测集、知识库和请求链路是否正确；是否实际记录了 contexts | 先验证数据与检索路径，再优化检索器，最后调整生成器 |

通常应遵循“先检索、后生成”的顺序：生成器无法从未被检索到的证据中产生可靠答案。不过，当上下文已经充分且忠实度仍低时，应该直接检查提示词、模型能力和回答约束，而不是继续无差别扩大召回范围。

---

## 10. 历史实验快照：如何正确解读

原始实验在相同汽车手册与少量问题上记录过不同 `chunk_size` 的得分，示例见下表：

| 指标 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.8667 | 0.9744 | 0.9915 |
| Answer Relevancy | 0.6068 | 0.8047 | 0.7200 |
| Context Recall | 0.3796 | 0.5000 | 1.0000 |
| Context Precision | 0.5849 | 0.5003 | 0.7344 |
| 检索 F1 | 0.4604 | 0.5001 | 0.8469 |

这些数字应视为**历史运行快照**，而不是“512 永远最优”的结论。它们会受到以下因素影响：

- 对话模型、embedding 模型、评测模型及其版本；
- Ragas、LangChain 和向量库版本；
- 问题数量与参考答案质量；
- `chunk_size`、overlap、Top-K、召回器权重和提示词；
- 模型输出与 LLM-as-a-judge 的随机性。

同样，曾有实验对 128 分块尝试混合检索、上下文压缩和“混合 + 重排序”。它们的价值在于提出可验证的假设：混合检索可能提升关键词覆盖，压缩可能降低上下文噪声，重排序可能提升相关内容的排名。请使用本章脚本、固定数据集和当前配置重新运行，再决定哪一种适合自己的业务。

---

## 11. 建立可持续的评估闭环

一次评测的正确打开方式不是挑出最高分，而是形成可重复的实验闭环：

1. **固定基线**：保存问题、参考答案、知识库版本、模型版本和当前配置；
2. **定义业务优先级**：医疗、法律等场景优先关注忠实度与召回，客服场景常更重视答案相关性和精度；
3. **每次只改变一个变量**：例如仅调整 `chunk_size`，不要同时更换模型、Top-K 和提示词；
4. **记录均值与逐题结果**：均值用于比较方案，逐题结果用于定位失败模式；
5. **人工抽样审核**：特别检查高分但实际无用、低分但评分误判的样本；
6. **先修证据链路**：优先处理知识库、分块、召回和排序问题，再优化提示词或模型；
7. **持续回归**：知识库、模型或检索配置变更后重新运行固定评测集。

Ragas 提供的是量化观察 RAG 的工具，不是替代领域判断的裁判。把指标、真实用户反馈和人工复核结合起来，才能让检索优化真正提升系统的可靠性。

---

## 12. 本章文件索引

| 文件 | 用途 |
| --- | --- |
| [`medical_evaluation.py`](medical_evaluation.py) | 对预置问答样本进行静态 Ragas 评测 |
| [`chunk_size_evaluation.py`](chunk_size_evaluation.py) | 比较不同分块大小下的纯 FAISS 检索效果 |
| [`automobile_handbook.py`](automobile_handbook.py) | 混合检索与 LLM 上下文压缩实验 |
| [`automobile_handbook_rebank.py`](automobile_handbook_rebank.py) | 混合检索与 OpenRouter 兼容重排序实验 |
| [`../langchaindemo/model.py`](../langchaindemo/model.py) | 对话模型、embedding 与重排序器工厂 |

下一步可以继续基于这套固定评测集，对查询改写、HyDE、多路召回、重排序和上下文压缩等 Advanced RAG 策略进行对照实验。
