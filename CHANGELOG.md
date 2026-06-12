# 更新日志

## [1.0.0] - 2025-01-10

### 新增
- RAG 核心引擎，支持文档检索和智能问答
- 支持 PDF/DOCX/TXT 格式文档解析
- MMR 检索算法，平衡相似度和多样性
- 幻觉抑制机制，无相关内容时拒答
- 440 条标注测试集（准确/模糊/干扰三类）
- Web API 服务（FastAPI）
- 单元测试和集成测试
- GitHub Actions CI 自动化测试
- Docker 部署配置
- 日志系统
- 监控统计模块

### 技术栈
- Python 3.12
- LangChain 1.3.4
- ChromaDB 1.5.9
- BGE-Small-ZH 嵌入模型
- DeepSeek-R1-Distill-Qwen-7b
- FastAPI 0.136.3
