# 后检索优化——RAG Fusion代码示例
from json import dumps, loads

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import LineListOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchaindemo.model import getModel, getEmbedding

model = getModel()
embedding_model = getEmbedding()


# 格式化输出文档内容
def format_print_docs(docs):
    for i, doc in enumerate(docs):
        print("-" * 100)
        print(f"Document {i}:\n{doc.page_content}")


def reciprocal_rank_fusion(results: list[list], k=60):
    """互逆排序融合算法，用于合并多个排序文档列表
    :param results: 包含多个排序文档列表的二维列表
    :param k: 融合公式中的平滑参数（默认60），值越小排名影响越大

    RRF分数计算：
    假设我们有两个查询检索结果：
    results = [
    [doc1, doc2, doc3],  # 第一个查询结果
    [doc2, doc4, doc1],  # 第二个查询结果
    ]
    k = 60

    在遍历过程中，doc1 在第一个查询中排名第1，在第二个查询中排名第3，它的 RRF 分数计算如下：
    第一个查询：1/(1+60) = 1/61
    第二个查询：1/(3+60) = 1/63
    累计分数：1/61 + 1/63

    通过这种方式，每个文档都会根据其在各个查询结果中的排名获取一个综合的分数，
    最终根据分数进行降序排序，得到融合后的结果。

    :return 按融合分数降序排列的文档列表，每个元素为(文档对象,分数)元组
    """

    # 初始化融合分数字典(key=序列化文档，value=累计分数)
    # 必须用dict而不能是list：同一个文档可能被多个查询检索到，需要按文档key累加分数
    fused_scores = {}
    # 遍历每个检索结果列表（每个查询对应的结果）
    for doc_list in results:
        # 对当前结果列表中的文档进行遍历（rank从1开始计算）
        for rank, doc in enumerate(doc_list, 1):
            # Document对象无法被标准库json直接序列化（会抛出TypeError），
            # 先用model_dump()转成纯字典再序列化，作为文档在字典中的唯一标识
            doc_str = dumps(doc.model_dump(), ensure_ascii=False)
            # 初始化文档得分（如果是首次出现）
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 计算并累加RRF分数：1/(当前排名+k)
            # 排名越靠前(rank值越小)的文档获得的分数越高
            fused_scores[doc_str] += 1 / (rank + k)

    # 按融合分数降序排序生成按相关度从高到低排列的文档列表。
    # sorted对fused_scores字典进行排序
    # fused_scores.items()返回键值对列表[(doc_str1, score1), (doc_str2, score2), ....]
    # key = lambda x: x[1]指定按分数(元组的第二个元素)排序
    # reverse = True 降序排序(分数高的在前)
    reranked_results = [
        (Document(**loads(doc)), score)  # 反序列化还原文档对象

        for doc, score in sorted(fused_scores.items(),
                                 key=lambda x: x[1],  # 取分数
                                 reverse=True)
    ]
    return reranked_results


# 加载文档
loader = TextLoader("deepseek.md", encoding="utf-8")
docs = loader.load()

# 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 创建向量数据库
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embedding_model)

# 创建检索器（k=3：每路查询各取3个候选文档）
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

original_query = "deepseek的应用场景"

# 创建prompt模板
prompt = ChatPromptTemplate.from_template("""
请根据下面给出的上下文来回答问题：
{context}

问题：{question}
""")

print("--------------------原始检索（单路查询）--------------------")
original_docs = retriever.invoke(original_query)
print("检索器检索的文档数量为：", len(original_docs))
format_print_docs(original_docs)

print("--------------------优化前回答--------------------")
chain = RunnableMap({
    "context": lambda x: "\n\n".join(doc.page_content for doc in original_docs),
    "question": lambda x: x["question"],
}) | prompt | model | StrOutputParser()
print(chain.invoke({"question": original_query}))

print("--------------------开始RAG-Fusion优化检索--------------------")
# RAG-Fusion = 多路查询生成（Multi-Query）+ RRF融合排序，再把TopK文档喂给LLM生成最终答案
# 与multi_query.py相比，多路召回的汇总方式从"直接去重合并"升级为"按排名融合排序"

# 第一步：让大模型基于原始问题生成4个不同角度的查询
generate_queries_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    你是一名AI语言模型助理。你的任务是针对原始问题生成4个不同角度的查询，
    用于从向量数据库中检索相关文档，通过多个视角的查询来克服基于距离的相似性搜索的局限性。
    请提供这些用换行符分隔的查询，不需要额外内容。
    原始问题：{question}
    """,
)

# LineListOutputParser的作用是将模型输出的文本按换行符分割成字符串列表
generate_queries_chain = generate_queries_prompt | model | LineListOutputParser()

print("--------------------第一步：生成多路查询--------------------")
queries = generate_queries_chain.invoke({"question": original_query})
for q in queries:
    print(q)

# 第二步：多路检索
# retriever.map()的作用是根据generate_queries的结果映射出N个retriever
# （可以理解为同时复制出N个retriever），与生成的N个query一一对应，
# 为每个query检索出一组相关文档（每组k=3个），4个query总共返回4组文档
print("--------------------第二步：多路检索--------------------")
retrieved_results = retriever.map().invoke(queries)
for i, (query, doc_list) in enumerate(zip(queries, retrieved_results)):
    print(f"查询{i + 1}：{query}")
    for j, doc in enumerate(doc_list):
        print(f"  文档{j + 1}: {doc.page_content.strip()[:60]}")

# 第三步：RRF融合重排
# 上面三步用LCEL也可以串成一条chain：
# fusion_chain = generate_queries_chain | retriever.map() | reciprocal_rank_fusion
print("--------------------第三步：RRF融合重排--------------------")
fused_results = reciprocal_rank_fusion(retrieved_results)

# 提取文档内容和对应分数，查看融合后的排序效果
contents = [doc[0].page_content for doc in fused_results]
scores = [doc[1] for doc in fused_results]

print("--" * 15, "最相关的文档及其得分：")
for content, score in zip(contents, scores):
    print(f"score={score:.4f} | {content.strip()[:60]}")

# 第四步：把融合后的TopK文档作为上下文喂给LLM，生成最终答案
# 多路查询会捞回大量候选（本例最多4×3=12个），并非所有文档都与问题相关，
# RRF按排名融合后，相关度最高的文档会排到最前面，只取TopK个即可
top_k = 3
top_docs = fused_results[:top_k]

final_chain = RunnableMap({
    "context": lambda x: "\n\n".join(doc[0].page_content for doc in top_docs),
    "question": lambda x: x["question"],
}) | prompt | model | StrOutputParser()

print("--------------------优化后回答--------------------")
print(final_chain.invoke({"question": original_query}))
