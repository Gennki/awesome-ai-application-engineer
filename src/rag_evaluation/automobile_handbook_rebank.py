import os

from datasets import Dataset
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from src.langchaindemo.model import getModel, getEmbedding, getReranker

# 创建模型客户端
llm = getModel()
embeddings = getEmbedding()
vllm = LangchainLLMWrapper(llm)
vllm_e = LangchainEmbeddingsWrapper(embeddings)
vectordb = None

# 加载文档
file_path = "data/初赛训练数据集.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print("文档个数：", len(docs))
# 分割文档
chunk_size = 128
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=int(chunk_size * 0.20),
)
split_docs = text_splitter.split_documents(docs)
print("分块个数：", len(split_docs))

index_folder_path = "data/faiss_index"
index_name = "c_default_" + str(chunk_size)
index_file_path = os.path.join(index_folder_path, f"{index_name}.faiss")
# 检查索引文件是否存在
if os.path.exists(index_file_path):
    print("索引文件已存在，直接加载...")
    vectordb = FAISS.load_local(index_folder_path, embeddings, index_name, allow_dangerous_deserialization=True)
else:
    print("索引文件不存在，创建并保存索引...")
    # 创建向量存储
    vectordb = FAISS.from_documents(split_docs, embeddings)
    # 保存索引
    vectordb.save_local(index_folder_path, index_name)
    print("向量化完成....")

'''
初始为10
检索器=mix_retriever<0.2,0.8>，指标为{'faithfulness': 0.9848, 'answer_relevancy': 0.9160, 
'context_recall': 1.0000, 'context_precision': 0.4811}  ，f1分数： 0.6497
检索器=faiss_retriever，指标为 {'faithfulness': 0.9650, 'answer_relevancy': 0.8994, 
'context_recall': 0.8519, 'context_precision': 0.9044}  ，f1分数： 0.8773
检索器=compression_retriever，指标为{'faithfulness': 0.9770, 'answer_relevancy': 0.8227, 
'context_recall': 0.7037, 'context_precision': 0.8057}  ，f1分数： 0.7513 
'''
topK_doc_count = 10

questions = ["如何使用安全带？", "车辆如何保养？", "座椅太热怎么办？"]
ground_truths = [
    '''调节座椅到合适位置，缓慢拉出安全带，将锁舌插入锁扣中，直到听见“咔哒”声。
    使腰部安全带应尽可能低的横跨于胯部。确保肩部安全带斜跨整个肩部，穿过胸部。
    将前排座椅安全带高度调整至合适的位置。
    请勿将座椅靠背太过向后倾斜。
    请在系紧安全带前检查锁扣插口是否存在异物（如：食物残渣等），若存在异物请及时取出。
    为确保安全带正常工作，请务必将安全带插入与之匹配的锁扣中。
    乘坐时，安全带必须拉紧，防止松垮，并确保其牢固贴身，无扭曲。
    切勿将安全带从您的后背绕过、从您的胳膊下面绕过或绕过您的颈部。安全带应远离您的面部和颈部，但不得从肩部滑落。
    如果安全带无法正常使用，请联系Lynk & Co领克中心进行处理。''',
    "为了保持车辆处于最佳状态，建议您定期关注车辆状态，包括定期保养、洗车、内部清洁、外部清洁、轮胎的保养、低压蓄电池的保养等。",
    '''有三种方式：1、通过中央显示屏，设置座椅加热强度或关闭座椅加热功能，
    在中央显示屏中点击座椅进入座椅加热控制界面，可在“关-低-中-高”之间循环。
    2、登录Lynk & Co App，按下前排座椅加热图标图标可以打开/关闭前排座椅加热。
    3、在中央显示屏中唤起空调控制界面然后点击舒适选项，降低座椅加热时间。'''
]

system_prompt = """
您是问答任务的助理。使用以下的上下文来回答问题，
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


class QAEvaluator:
    def __init__(self, retriever):
        # 创建文档链
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        # 创建检索链
        self.chain = create_retrieval_chain(retriever, document_chain)
        self.retriever = retriever

    def generate_answers(self, questions):
        answers = []
        contexts = []
        for question in questions:
            print("问题：", question)
            response = self.chain.invoke({"input": question})
            print("大模型答复：", response["answer"], "\n")
            answers.append(response["answer"])
            # 获取上下文
            contexts.append([doc.page_content for doc in response["context"]])
            print("大模型回答时参考的上下文：", contexts, "\n")
            print("==" * 35)
        return answers, contexts

    def evaluate(self, questions, answers, contexts, ground_truths):
        evaluate_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        evaluate_dataset = Dataset.from_dict(evaluate_data)
        evaluate_result = evaluate(
            evaluate_dataset,
            llm=vllm,
            embeddings=vllm_e,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]
        )
        return evaluate_result


def exec_eval(retriever):
    qa_evaluator = QAEvaluator(retriever)
    answers, contexts = qa_evaluator.generate_answers(questions)
    return qa_evaluator.evaluate(questions, answers, contexts, ground_truths)


def calc_f1(evaluate_result):
    context_precisions = evaluate_result["context_precision"]
    context_recalls = evaluate_result["context_recall"]
    print("context_precisions=", context_precisions)
    print("context_recalls=", context_recalls)
    context_precision_score = sum(context_precisions) / len(context_precisions)
    context_recall_score = sum(context_recalls) / len(context_recalls)
    f1_score = (2 * context_precision_score * context_recall_score) / (context_precision_score + context_recall_score)
    return round(f1_score, 4)


# 创建向量检索器，擅长根据语义相似度查找相关文档
faiss_retriever = vectordb.as_retriever(search_kwargs={"k": topK_doc_count})
# 创建全文检索器，擅长根据全文查找相关文档
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = topK_doc_count
# 创建混合检索器
mix_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weight=[0.2, 0.8])
rerank_retriever = getReranker()
combin_retriever = ContextualCompressionRetriever(
    base_compressor=rerank_retriever, base_retriever=mix_retriever)
mix_evaluate_result = exec_eval(combin_retriever)
print("重排序后mix_retriever评估结果：", mix_evaluate_result, " ，f1分数：", calc_f1(mix_evaluate_result))
