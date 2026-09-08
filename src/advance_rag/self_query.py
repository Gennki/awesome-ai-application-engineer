# 元数据索引代码示例
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.base import get_query_constructor_prompt, StructuredQueryOutputParser
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_core.documents import Document

from src.langchaindemo.model import getEmbedding, getModel

# 加载文档
docs = [
    # 1. 人工智能与自动驾驶 (2024)
    Document(
        page_content="百度发布Apollo开放平台10.0，通过零拷贝通信实现微秒级传输，性能提升10倍，支撑L4级自动驾驶在单Orin芯片上稳定落地",
        metadata={"year": 2024, "rating": 9.2, "genre": "AI/自动驾驶", "author": "百度"}
    ),
    # 2. 区块链与跨境贸易结算 (2023)
    Document(
        page_content="本钢集团通过欧冶金服EFFITRADE平台完成首笔跨境区块链铁矿石交易，实现信用证点对点对接，资金占用时间缩短约3天",
        metadata={"year": 2023, "rating": 9.8, "genre": "区块链/贸易", "author": "本钢集团"}
    ),
    # 3. 云计算与量子计算模拟 (2022)
    Document(
        page_content="富士通开发出世界最快的模拟量子计算机，在PRIMEHPC FX 700集群上成功处理36个量子比特的电路，目标2022年9月前开发40量子比特模拟器",
        metadata={"year": 2022, "rating": 8.6, "genre": "云计算/量子", "author": "富士通"}
    ),
    # 4. 强化学习与工业机器人 (2024)
    Document(
        page_content="2024年多项研究聚焦于基于深度强化学习(DRL)的工业机器人节能轨迹规划，旨在显著降低轨迹生成时间与能耗",
        metadata={"year": 2024, "rating": 8.9, "genre": "AI/机器人", "author": "IEEE研究组"}
    ),
    # 5. 区块链与医疗数据共享 (2023)
    Document(
        page_content="2023年研究提出基于区块链的医疗数据共享模型，优化后总执行时间较传统区块链模型减少约49%，数据下载延迟可低至200ms",
        metadata={"year": 2023, "rating": 9.5, "genre": "区块链/医疗", "author": "OUCI团队"}
    ),
    # 6. 5G边缘计算与视频分析 (2022)
    Document(
        page_content="2022年提出的Tutti系统将5G RAN与移动边缘计算(MEC)结合，用于延迟关键型视频分析，平均响应延迟比现有5G MEC系统降低61.69%",
        metadata={"year": 2022, "rating": 8.3, "genre": "5G/边缘计算", "author": "ACM MobiCom"}
    ),
    # 7. 生成式AI与药物发现 (2024)
    Document(
        page_content="2024年研究提出基于生成式AI的药物设计模型ClickGen，成功发现两个对PARP1酶具有纳摩尔级抑制活性的先导化合物",
        metadata={"year": 2024, "rating": 9.7, "genre": "AI/药物发现", "author": "Nature Communications"}
    ),
    # 8. 跨链互操作协议 (2023)
    Document(
        page_content="2023年摩根大通Onyx与Axelar合作，利用跨链技术实现与Provenance区块链的互操作性，该链锁定超过90亿美元的真实世界资产",
        metadata={"year": 2023, "rating": 9.1, "genre": "区块链/跨链", "author": "摩根大通"}
    ),
    # 9. 云原生数据库HTAP (2022)
    Document(
        page_content="国产分布式数据库OceanBase在2022年入选Forrester Translytical报告，是全球唯一在TPC-C和TPC-H测试中都刷新世界纪录的数据库",
        metadata={"year": 2022, "rating": 8.8, "genre": "云计算/数据库", "author": "OceanBase团队"}
    ),
    # 10. 大语言模型与法律 (2024)
    Document(
        page_content="2024年研究显示，在法律文书生成任务中，专业微调模型SaulLM在少样本提示下精确度达81%，显著优于通用模型",
        metadata={"year": 2024, "rating": 9.4, "genre": "AI/法律", "author": "SaulLM研究组"}
    ),
    # 11. 零知识证明与匿名投票 (2023)
    Document(
        page_content="2023年研究提出基于零知识证明的智能合约投票系统，通过交互式证明协议确保投票者匿名且投票内容可验证",
        metadata={"year": 2023, "rating": 9.6, "genre": "区块链/隐私", "author": "工程科学学报"}
    ),
    # 12. 无服务器架构大数据处理 (2022)
    Document(
        page_content="2022年Amazon EMR Serverless正式推出，使数据工程师无需管理集群即可在云中运行PB级Spark、Hive等大数据分析任务",
        metadata={"year": 2022, "rating": 8.5, "genre": "云计算/大数据", "author": "AWS"}
    )
]

vectorstore = Chroma.from_documents(docs, getEmbedding())

# 元数据字段定义（指导LLM如何解析查询条件）
meta_field_info = [
    AttributeInfo(
        name="year",
        description="技术成果或研究发表的年份，整数，范围为 2022-2024",
        type="integer"
    ),
    AttributeInfo(
        name="rating",
        description="技术价值或影响力评分（浮点数，范围 0.0-10.0），分数越高代表业界认可度越高",
        type="float"
    ),
    AttributeInfo(
        name="genre",
        description="技术领域分类，例如 'AI/自动驾驶'、'区块链/贸易'、'云计算/量子'、'5G/边缘计算'、'AI/药物发现'、'区块链/隐私' 等",
        type="string"
    ),
    AttributeInfo(
        name="author",
        description="发布该技术的公司、团队或主要研究机构（如百度、富士通、摩根大通、AWS 等）",
        type="string"
    )
]

# 文档内容描述（指导LLM理解文档内容）
document_content_description = "技术文章简述"

# 创建自查询检索器（核心组件）
'''
SelfQueryRetriever 是 langchain 库中的一个工具,其主要功能是把自然语言查询转变为结构化查询,
以此提升检索的精准度。它整合了检索器和语言模型,能依据查询内容自动推断出筛选条件,还能识别出相关的元数据字段。 
'''
retriever = SelfQueryRetriever.from_llm(
    getModel(),
    vectorstore,
    document_content_description,
    meta_field_info,
    # enable_limit=True, # 限定只返回一个结果
)

# print(retriever.invoke("2024年且评分在9分以上的文章"))
print(retriever.invoke("百度发布的文章"))


# # 原理：构建查询解析器（分析内部工作机制）
# """构建查询提示模板
# document_content_description：对文档内容的概括性描述，例如“有关各种主题的文章”
# meta_field_info：元数据字段的详细描述，涵盖字段名称、类型和描述
# 此函数会生成一个提示模板，其用途是指导语言模型如何将自然语言查询转换为结构化查询
# """
# prompt = get_query_constructor_prompt(
#     document_content_description,
#     meta_field_info
# )
#
# # 解析器的作用是把语言模型的输出转换为 StructuredQuery 结构化查询对象
# # 这个对象包含了 query（文本查询）和 filter（元数据筛选条件）
# output_parser = StructuredQueryOutputParser.from_components()
# # 链式操作，先将用户查询填入提示词模板，接着由语言模型生成结构化输出，最后沟通过解析器得到结构化查询
# query_constructor_chain = prompt | getModel() | output_parser
#
# # 打印查询构造提示
# print("提示词：", prompt.format(query="检索2023年发布的文章"))
# print("---------------------------------------------------")
# print("结构化查询结果：", query_constructor_chain.invoke({
#     "query": "检索2023年发布的文章"
# }))
