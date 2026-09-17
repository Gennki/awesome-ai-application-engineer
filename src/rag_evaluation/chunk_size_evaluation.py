# 应用实战2，评估选择合适的知识块大小
import os.path

from datasets import Dataset
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import evaluate, RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from src.langchaindemo.model import getEmbedding, getModel

model = getModel()
embeddings = getEmbedding()
vllm = LangchainLLMWrapper(model)
vllm_e = LangchainEmbeddingsWrapper(embeddings)

# 加载PDF文档
file_path = "data/初赛训练数据集.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print("文档个数：", len(docs))

# 分割文档
# 分别测试chunk_size = 128,256,512，chunk_overlap为一般chunk_size的1/10~1/4，常用的1/5，维持topK_doc_count = 10
'''
chunk_size = 128 ：
默认分块分割符下，指标为 {'faithfulness': 0.9650, 'answer_relevancy': 0.8994, 
'context_recall': 0.8519, 'context_precision': 0.9044}  ，f1分数： 0.8773
调整分块分割符下，指标为 {'faithfulness': 0.7857, 'answer_relevancy': 0.9101,
'context_recall': 0.8333, 'context_precision': 0.3301}'''
'''
chunk_size = 256：
默认分块分割符下，指标为{'faithfulness': 0.9697, 'answer_relevancy': 0.8851, 
'context_recall': 0.8889, 'context_precision': 0.7775}  ，f1分数： 0.8295
调整分块分割符下，指标为{'faithfulness': 0.7315, 'answer_relevancy': 0.9136, 
'context_recall': 1.0000, 'context_precision': 0.3869}
'''
'''
chunk_size = 512：
默认分块分割符下，指标为{'faithfulness': 0.9570, 'answer_relevancy': 0.9301, 
'context_recall': 0.8889, 'context_precision': 0.7680}  ，f1分数： 0.8241
调整分块分割符下，指标为{'faithfulness': 0.6000, 'answer_relevancy': 0.8851,
'context_recall': 1.0000, 'context_precision': 0.3213}'''
chunk_size = 512
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.2))
split_docs = text_splitter.split_documents(docs)
print("分块个数：", len(split_docs))

# 创建向量存储
index_folder_path = "data/faiss_index"
index_name = "c_default_" + str(chunk_size)
index_file_path = os.path.join(index_folder_path, f"{index_name}.faiss")
# 检查索引文件是否存在
if os.path.exists(index_file_path):
    print("索引文件已存在，直接加载...")
    vectordb = FAISS.load_local(folder_path=index_folder_path,
                                embeddings=embeddings,
                                index_name=index_name,
                                allow_dangerous_deserialization=True)
else:
    print("索引文件不存在，创建并保存索引...")
    vectordb = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    vectordb.save_local(folder_path=index_folder_path, index_name=index_name)
    print("向量化完成...")

topK_doc_count = 10
questions = ["如何使用安全带？", "车辆如何保养？", "座椅太热桌面版？"]
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
你是问答任务助理。使用以下的上下文来回答问题。
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{input}")
])


class QAEvaluator:
    def __init__(self, retriever):
        # 创建文档链
        document_chain = create_stuff_documents_chain(model, prompt_template)
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
                context_precision
            ]
        )
        return evaluate_result


def exec_evaluate(retriever):
    qa_evaluator = QAEvaluator(retriever)
    answers, contexts = qa_evaluator.generate_answers(questions=questions)
    return qa_evaluator.evaluate(questions=questions, answers=answers, contexts=contexts, ground_truths=ground_truths)


# 计算F1分数
def calc_f1(evaluate_result):
    context_precision = evaluate_result["context_precision"]
    context_recall = evaluate_result["context_recall"]
    print("context_precision=", context_precision)
    print("context_recall=", context_recall)
    context_precision_score = sum(context_precision) / len(context_precision)
    context_recall_score = sum(context_recall) / len(context_recall)
    f1_score = 2 * (context_precision_score * context_recall_score) / (context_precision_score + context_recall_score)
    return round(f1_score, 4)


faiss_retriever = vectordb.as_retriever(search_kwargs={"k": topK_doc_count})
evaluate_result = exec_evaluate(faiss_retriever)
print("chunk_size=[", chunk_size, "]评估结果：", evaluate_result, " ，f1分数：", calc_f1(evaluate_result))
