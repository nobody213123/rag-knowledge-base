# 技术设计文档

## 技术栈
| 组件 | 选型 | 版本 |
|------|------|------|
| 编程语言 | Python | 3.12 |
| RAG 框架 | LangChain | 1.3.4 |
| 向量数据库 | ChromaDB | 1.5.9 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | - |
| 大语言模型 | DeepSeek-R1-Distill-Qwen-7b | - |
| Web 框架 | FastAPI | 0.136.3 |
| LLM 客户端 | OpenAI SDK | 2.41.0 |

## 技术选型理由

### 为什么选 ChromaDB 而不是 FAISS？
- ChromaDB 是专用向量数据库，自带持久化、元数据过滤
- FAISS 更适合纯向量检索场景，需要自己管理存储
- ChromaDB 更适合中小型项目，部署简单

### 为什么选 DeepSeek-R1 而不是 GPT-4？
- DeepSeek-R1 推理能力强，适合知识库问答
- 成本远低于 GPT-4
- 阿里云百炼平台国内访问稳定

### 为什么选 BGE-Small-ZH 而不是 M3E？
- BGE 系列在中文评测榜单表现优秀
- Small 版本轻量（约 100MB），速度快
- HuggingFace 生态支持好

### 为什么用 MMR 检索而不是普通相似度检索？
- 普通检索返回结果可能高度重复
- MMR 平衡相似度和多样性，提升信息覆盖度

## 系统架构

```
用户界面层    Web API（FastAPI）
    ↓
编排层        RAG 引擎（LangChain）
    ↓
模型层        Embedding（BGE）+ LLM（DeepSeek）
    ↓
工具层        文档加载器、文本分割器
    ↓
存储层        ChromaDB 向量库
    ↓
监控层        日志系统 + 调用统计
```

## 数据流

### 构建阶段
```
文档(PDF/DOCX/TXT)
  → Loader 加载
  → RecursiveCharacterTextSplitter 切块(200字/50重叠)
  → HuggingFaceEmbeddings 向量化(512维)
  → ChromaDB 持久化存储
```

### 查询阶段
```
用户问题
  → Embeddings 向量化
  → ChromaDB MMR 检索(k=10, fetch_k=20)
  → 拼接上下文
  → Prompt 模板(带拒答规则)
  → DeepSeek-R1 生成回答
  → 返回答案 + 来源 + 耗时
```

## API 设计

| 接口 | 方法 | 说明 |
|------|------|------|
| `/ask` | POST | 提问，返回答案和来源 |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 调用统计 |
| `/rebuild` | POST | 重建索引 |
| `/docs` | GET | Swagger API 文档 |
