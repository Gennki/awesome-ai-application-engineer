"""Connect to MySQL and run a query through LangChain's SQL utilities."""

from __future__ import annotations

import os
import re
from operator import itemgetter
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.langchaindemo.model import getModel


# 创建数据库实例
def create_database() -> SQLDatabase:
    """Create a SQLDatabase instance from the MYSQL_* environment variables."""
    load_dotenv()

    password = os.getenv("MYSQL_PASSWORD")
    if not password:
        raise ValueError("MYSQL_PASSWORD is required")

    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    database = os.getenv("MYSQL_DATABASE", "rag")

    # Quote credentials so passwords containing symbols remain valid in the URI.
    uri = (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(database)}?charset=utf8mb4"
    )

    return SQLDatabase.from_uri(uri)


def clean_sql_query(sql_text: str) -> str:
    """Extract a SQL statement from the model output."""
    marker = "SQLQuery:"
    if marker in sql_text:
        sql_text = sql_text.split(marker, 1)[1]
    sql_text = sql_text.strip()
    if sql_text.startswith("```"):
        sql_text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql_text).strip()
    return sql_text


def validate_read_only_sql(sql: str) -> str:
    """Allow one read-only SELECT statement, optionally preceded by a CTE."""
    statement = sql.strip()
    if not statement:
        raise ValueError("生成的 SQL 不能为空")
    if ";" in statement.rstrip(";"):
        raise ValueError("只允许执行单条 SQL")

    normalized = statement.rstrip(";").lstrip().lower()
    if not (normalized.startswith("select") or normalized.startswith("with")):
        raise ValueError("只允许执行 SELECT 或 WITH ... SELECT 查询")

    forbidden = re.compile(
        r"\b(insert|update|delete|replace|merge|create|alter|drop|truncate|grant|revoke|call|load|outfile|dumpfile|set|use|lock|unlock)\b",
        re.IGNORECASE,
    )
    if forbidden.search(normalized):
        raise ValueError("SQL 包含不允许的写入或管理操作")
    return statement


def main() -> None:
    db = create_database()
    model = getModel()
    sql_make_chain = (
        create_sql_query_chain(model, db) | RunnableLambda(clean_sql_query)
    )
    answer_prompt = PromptTemplate.from_template("""
给定以下用户问题、可能的SQL语句和SQL执行后的结果，回答用户问题：
Question: {question}
SQL Query: {query}
SQL Result: {result}
回答：
""")
    execute_sql_tool = QuerySQLDataBaseTool(db=db)

    def execute_read_only_sql(sql: str) -> str:
        return execute_sql_tool.invoke(validate_read_only_sql(sql))

    chain = (
        RunnablePassthrough.assign(query=sql_make_chain)
        .assign(result=itemgetter("query") | RunnableLambda(execute_read_only_sql))
        | answer_prompt
        | model
        | StrOutputParser()
    )

    question = "请从国家表中查询出China的相关数据"
    generated_sql = sql_make_chain.invoke({"question": question})
    print("生成的 SQL：", validate_read_only_sql(generated_sql))
    print("=" * 100)
    print("最终执行的结果：", chain.invoke({"question": question}))


if __name__ == "__main__":
    main()
