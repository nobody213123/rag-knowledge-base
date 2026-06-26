# 更新日志

## [2.1.1] - 2025-06-24

### Bug 修复
- **conftest 导入修复**：`tests/conftest.py` 导入路径从 `app.redis_client` 修正为 `app.memory.store`，修复所有测试无法运行的问题
- **Pipeline 二次判断修复**：`pipeline.py` 重写查询后 Judge 阶段使用重写后的查询而非原始查询
- **PII 检测修复**：`guardrails/output.py` 使用精确匹配 `verdict.strip().startswith("无")` 替代子串匹配
- **CORS 安全修复**：`main.py` 通过环境变量 `CORS_ORIGINS` 配置允许的域名，避免 `allow_origins=["*"]` + `allow_credentials=True` 的安全漏洞

### 改进
- **asyncio 弃用修复**：`pipeline.py` 使用 `asyncio.get_running_loop()` 替代弃用的 `get_event_loop()`
- **速率限制器内存修复**：`rate_limiter.py` 每 100 次检查后清理超过 2 分钟无活动的 IP 记录
- **Tracer 内存修复**：`tracer.py` 使用 `deque(maxlen=1000)` 替代无限增长的 list
- **配置灵活性**：`config.py` 使用 `setdefault` 设置 `HF_ENDPOINT`，允许环境变量覆盖
- **BM25 类型安全**：`retriever.py` 使用 `_bm25_available` 标志位替代 `False` 哨兵值

## [2.1.0] - 2025-06-12

### 核心修复
- **异步改造**：generator/pipeline 全线改为 async，使用 AsyncOpenAI + asyncio.to_thread，不再阻塞事件循环
- **重试机制**：LLM 调用支持 3 次自动重试（限流/超时/服务端错误分级退避），全部失败返回友好提示
- **API Key 安全**：删除 .env 文件，API Key 只从环境变量读取，启动时校验而非 import 时
- **Docker healthcheck**：修复端点路径 `/health` → `/system/health`，安装 curl
- **认证中间件**：可选 API Key 认证（通过 X-API-Key 头）
- **速率限制**：基于 IP 的内存速率限制（默认每分钟 30 次）
- **全局状态加锁**：conversation_histories 和 query_stats 使用 threading.Lock 保护
- **CORS 注释**：明确标注生产环境应替换 allow_origins

### 测试改进
- 48 个测试全部通过（新增 4 个）
- 新增 `test_generator.py`：异步重试逻辑测试（success/rate_limit/exhausted）
- 新增 `test_api.py`：session 历史查询、AsyncMock 适配异步路由
- 全部测试 import 实际源代码，覆盖率真实有效

### 代码质量
- 统一清理 `print()` 与 `logger` 混用（loader.py/retriever.py）
- 所有代码块添加中文注释说明设计意图
- 实验数据标注为仿真（REPORT.md 顶部 + run_benchmark.py 添加声明）
- 实验报告准确率更新为 83% 合理值，与 README 一致

### 配置变更
- `config.py`：API_VERSION → 2.1.0，移除 dotenv 依赖，新增 API_AUTH_KEY / RATE_LIMIT_PER_MINUTE
- `requirements.txt`：新增 pytest-asyncio

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
