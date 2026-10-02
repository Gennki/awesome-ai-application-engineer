import operator
import re
from typing import Annotated, Sequence

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from src.langchaindemo.model import getModel

load_dotenv()
# RetryPolicy()

db = SQLDatabase.from_uri("sqlite:///:memory:")
# 创建表
db.run("CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name NVARCHAR(120));")
# 表中添加数据
db.run("""
INSERT INTO Artist (Name) VALUES
('Louis Armstrong'),
('Duke Ellington'),
('Ella Fitzgerald'),
('Charlie Parker'),
('Thelonious Monk'),
('Billie Holiday'),
('Dizzy Gillespie'),
('Herbie Hancock'),
('Sarah Vaughan'),
('Chet Baker'),
('Dave Brubeck'),
('Stan Getz'),
('Art Tatum'),
('Coleman Hawkins'),
('Lester Young');
""")

model = getModel()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def query_database(state):
    # 提取纯SQL查询，去除Markdown标记
    raw_sql = state['messages'][1].content
    # 使用正则表达式提取SQL语句
    sql_match = re.search(r'(SELECT.*?;)', raw_sql, re.DOTALL | re.IGNORECASE)
    if sql_match:
        clean_sql = sql_match.group(1)
    else:
        # 如果没有找到明确的SQL语句，使用原始内容但去除常见的标记
        clean_sql = re.sub(r'```.*?\n', '', raw_sql).strip()
        clean_sql = re.sub(r'```', '', clean_sql).strip()

    print('执行的SQL:', clean_sql)
    query_result = db.run(clean_sql)
    print('query_result:', query_result)
    return {"messages": [AIMessage(content=query_result)]}


def call_model(state):
    response = model.invoke(state["messages"])
    print('response:', response)
    return {"messages": [response]}


builder = StateGraph(AgentState)
# 两种不同的重试策略，来自RetryPolicy类的字段
# builder.add_node("query_database",query_database,retry=RetryPolicy(retry_on=sqlite3.OperationalError))
builder.add_node("query_database", query_database)
builder.add_node("model", call_model, retry=RetryPolicy(max_attempts=5))

builder.add_edge(START, "model")
builder.add_edge("model", "query_database")
builder.add_edge("query_database", END)
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='../../../../assets/示例9.png')
# result= graph.invoke({"messages": [HumanMessage(content="查询Artist表格中前十位艺术家，只返回SQL查询语句，不要返回其他内容，特别是```sql\n不要出现")]})
result = graph.invoke({"messages": [HumanMessage(content="查询Artist表格中前十位艺术家")]})
print('result：', result)
