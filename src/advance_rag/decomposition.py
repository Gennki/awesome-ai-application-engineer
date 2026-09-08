# 查询优化-问题分解代码示例
from typing import List

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import LineListOutputParser
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate, BasePromptTemplate, ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from src.langchaindemo.model import getModel, getEmbedding

model = getModel()
embedding_model = getEmbedding()

# 创建示例文档数据集：番茄炒蛋的制作指南
documents = [
    Document(page_content="""
    番茄炒蛋的食材：番茄2个（中等大小，选熟透多汁的）、鸡蛋3个、小葱1根、盐适量、白糖半小勺、食用油适量。可选配料：白胡椒粉少许（去腥）、料酒几滴（去腥提香）、生抽1-2滴（提鲜）。
    """),
    Document(page_content="""
    番茄炒蛋的步骤：
    处理番茄：番茄顶部划十字，用开水烫10-30秒，撕去外皮后切小块。如果不介意口感，此步可省略。
    准备蛋液：鸡蛋打入碗中，加一小撮盐和几滴料酒或一勺清水，顺着一个方向充分打散至起泡。
    炒鸡蛋：热锅倒油，油热后倒入蛋液，待底部凝固后用锅铲快速划散成大块，约八成熟时立刻盛出备用。
    炒番茄：锅中留底油，倒入番茄块，中火翻炒并边炒边用铲背按压，炒至出汁变软。
    混合调味：将炒好的鸡蛋倒回锅中，加盐和白糖调味，快速翻炒均匀，让蛋块裹满番茄汁。
    出锅：关火，撒上葱花即可装盘。
    """),
    Document(page_content="""
    番茄炒蛋的技巧与注意事项：
    选材与备料：番茄选捏起来稍软的，更容易炒出浓汁。蛋液里加少许清水或料酒，炒出来更嫩滑且能去腥。
    火候与顺序：先炒蛋盛出，再炒番茄，最后混合，能保证鸡蛋嫩滑、番茄出汁充分。炒蛋用中大火，凝固即出锅；炒番茄用中火，避免炒干。
    关键调味：盐可以帮助番茄快速出汁，在炒番茄时先放；白糖用来中和番茄的酸味和提鲜，量不宜多，以吃不出明显甜味为宜。
    """)
]


# 格式化输出文档内容
def format_print_docs(docs):
    for i, doc in enumerate(docs):
        print("-" * 100)
        print(f"Document {i}:\n{doc.page_content}")


# format_print_docs(documents)

# 创建Chroma向量存储并添加文档
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, collection_name="decomposition")
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

print("------------------------检索到的文档（拆解前）------------------------")
# 从实际来说，该问题的答案，包括原材料、步骤，以及注意事项才算完整
format_print_docs(retriever.invoke("新手该如何制作番茄炒蛋？"))

print("------------------------开始问题拆解的处理------------------------")
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    你是一名AI语言模型助理。你的任务是将输入的问题分解成3个子问题，通过一个个解决这些子问题从而解决完整的问题。
    子问题需要在矢量数据库中检索相关文档。通过分解用户问题生成子问题，你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。
    请提供这些用换行符分割的子问题本身，不需要额外内容。
    原始问题：{question}
    """,
)

# print("------------------------测试大模型对问题的拆解，实际业务中可不用------------------------")
# # LineListOutputParser的作用是将模型输出的文本按换行符分割成字符串列表
# chain = DEFAULT_QUERY_PROMPT | model | LineListOutputParser()
# result = chain.invoke({"question": "新手该如何制作番茄炒蛋？"})
# print(result)
# print("------------------------完成测试大模型对问题的拆解------------------------")

# 子问题回答提示模板
DEFAULT_SUB_QUESTION_PROMPT = PromptTemplate(
    input_variables=["question", "sub_question", "documents"],
    template="""
    要解决主要问题{question}，需要先解决子问题{sub_question}。
    以下是未支持您的推理而提供的参考文档：{documents}。请直接给出当前子问题的答案。不需要额外内容。
    """,
)


# 自定义一个检索器，将对子问题的生成、获得子问题的答案组合起来，通过组合简化使用过程
# 继承自BaseRetriever
class DecompositionQueryRetriever(BaseRetriever):
    # 定义3个属性
    # 向量数据库检索器
    retriever: BaseRetriever  # 基础检索器，用于检索与子问题相关的文档
    # 生成子问题链
    make_sub_chain: Runnable
    # 解决子问题链
    resolve_sub_chain: Runnable

    # 构造函数
    @classmethod  # 类方法装饰器
    # self表示实例，clas表示类
    def from_llm(
            cls,
            retriever: BaseRetriever,  # 基础检索器
            llm: BaseLanguageModel,  # 语言模型
            prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,  # 子问题生成提示词模板
            sub_prompt: BasePromptTemplate = DEFAULT_SUB_QUESTION_PROMPT,  # 子问题回答提示词模板
    ) -> "DecompositionQueryRetriever":
        """类方法：通过LLM创建分解检索器实例"""
        output_parser = LineListOutputParser()
        return cls(
            retriever=retriever,
            make_sub_chain=prompt | llm | output_parser,
            resolve_sub_chain=sub_prompt | llm
        )

    # 主入口，协调整个分解流程
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # 生成子问题列表
        sub_queries = self.generate_queries(query)
        # 将子问题列表传递给 retrieve_documents 解决子问题并构建结果文档
        documents = self.retrieve_documents(query, sub_queries)
        return documents

    # 生成子问题，使用LLM将用户问题分解为子问题
    def generate_queries(self, question: str) -> List[str]:
        """
        输入一个交通问题。使用make_sub_chain生成子问题列表。
        它调用链并返回一个字符串列表，每个字符串是一个子问题。
        """
        response = self.make_sub_chain.invoke({"question": question})
        print(f"生成的子问题：{response}")
        return response

    # 获得子问题答案，并行处理子问题，检索+生成答案
    def retrieve_documents(self, query: str, sub_queries: List[str]) -> List[Document]:
        """
        输入原始查询（query）和子问题列表（sub_queries:List[str]），对每个子问题并行处理：
        使用retriever检索与子问题相关的文档
        使用resolve_sub_chain根据检索到的文档生成子问题的答案。
        然后，将每个子问题和答案组合成一个Document对象（page_content为子问题+答案）。
        最后返回这些文档列表
        """

        def process_sub_query(sub_query: str):
            # 检索子问题相关文档
            docs = self.retriever.invoke(sub_query)
            # 构建子问题解答提示词模板，并获取答案
            return self.resolve_sub_chain.invoke({
                "question": query,  # 原始问题
                "sub_question": sub_query,  # 当前子问题
                "documents": [doc.page_content for doc in docs]  # 检索到的文档内容
            })

        # 创建可运行的处理链
        sub_llm_chain = RunnableLambda(process_sub_query)
        # 批量执行所有的子问题
        responses = sub_llm_chain.batch(sub_queries)
        # 将子问题和答案合并作为解决主问题的文档
        documents = [
            Document(page_content=sub_query + "\n" + response.content)
            for sub_query, response in zip(sub_queries, responses)
        ]
        return documents


# ========================== 主执行流程 ==========================
print("使用DecompositionQueryRetriever来分解问题")
# 创建问题分解检索器实例
decompositionQueryRetriever = DecompositionQueryRetriever.from_llm(llm=model, retriever=retriever)
# 执行问题分解检索
decomposition_docs = decompositionQueryRetriever.invoke("新手该如何制作番茄炒蛋？")
print("--------------------------检索到的文档（拆解后）--------------------------")
format_print_docs(decomposition_docs)
# 创建prompt模板
prompt = ChatPromptTemplate.from_template("""
请根据以下文档回答问题：
### 文档：
{context}
### 问题：
{question}
""")
chain = prompt | model
print("--------------------------回答--------------------------")
response = chain.invoke(
    {"context": [doc.page_content for doc in decomposition_docs], "question": "新手该如何制作番茄炒蛋？"})
print(response.content)
