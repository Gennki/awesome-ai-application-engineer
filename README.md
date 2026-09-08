# awesome-ai-application-engineer

一个用于学习大语言模型应用开发的示例项目，按从基础概念到检索增强与 Agent 工程实践的顺序组织代码和教程。

## 学习路线

建议按以下顺序阅读：

1. [基础 RAG：从文档到答案](src/rag/01_基础RAG.md)：手动实现文档分块、向量化、向量检索和生成回答。
2. [LangChain 入门：把模型调用组装成 chain](src/langchaindemo/02_LangChain入门.md)：学习 Prompt、输出解析器和 LCEL chain 的基本用法。
3. [RAG 与 LangChain 实战：接入真实文档](src/rag_and_langchain_demo/03_RAG与LangChain实战.md)：使用 Loader、Document 和 retriever 接入 Word 文档与网页。
4. [Tool 与 Function Calling：模型提出请求，程序决定执行](src/tool/04_Tool与FunctionCall.md)：理解工具定义、工具调用循环和执行权限边界。
5. [Agent 中间件：在模型调用前后插入规则](src/middleware/05_Agent中间件.md)：在 Agent 的模型调用前后加入日志、脱敏、摘要和拦截规则。
6. [Advanced RAG：针对问题选择检索优化](src/advance_rag/06_Advanced%20RAG.md)：根据失败模式选择多路召回、混合检索、重排序和上下文压缩等优化方案。

## 目录结构

- `src/rag/`：手动实现的基础 RAG
- `src/langchaindemo/`：LangChain 基础组件与 LCEL 示例
- `src/rag_and_langchain_demo/`：基于 LangChain 的真实文档 RAG
- `src/tool/`：Tool 与 Function Calling 示例
- `src/middleware/`：Agent 中间件示例
- `src/advance_rag/`：Advanced RAG 检索优化示例

## 环境准备

安装项目依赖：

```bash
pip install -r requirements.txt
```

需要调用模型或 embedding 服务的示例，请根据 `.env.example` 配置对应的环境变量。各篇教程会说明运行方式、依赖服务和适用范围。
