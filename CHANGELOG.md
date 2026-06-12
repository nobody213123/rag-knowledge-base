# 更新日志

## [2.0.0] - 2025-06-12

### 重构
- 扁平结构 → 模块化 `app/` 架构（api/rag/evaluation/schemas）
- 新增 `scripts/` 命令行工具（build_index/run_eval/chat_cli）

### 新增
- 前端聊天界面（`frontend/`，原生 HTML+CSS+JS）
- 27 → 44 个测试用例（含 API 端点测试、pipeline 测试）
- CORS 支持（跨域访问）
- LLM 调用自动重试 + 错误降级机制
- 基于 session_id 的对话历史隔离
- Docker Compose 编排

### 修复
- Dockerfile 入口（`rag_engine.py` → `app.main`）
- 系统统计死代码（`record_query` 已接入端点）
- `print()` 与 `logger` 混用
- `.dockerignore` 添加排除规则

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
