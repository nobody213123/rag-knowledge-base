"""
全局配置模块
所有配置集中管理，避免到处找配置
"""
import os
from pathlib import Path

# ============================================================
# 重要：不要在本文件或任何代码中硬编码 API Key
# 通过环境变量 DASHSCOPE_API_KEY 注入
# 本地开发可在终端执行: export DASHSCOPE_API_KEY=sk-xxx
# 或使用 .env 文件（已加入 .gitignore，不会被提交）
# ============================================================

# HuggingFace 国内镜像（仅在中国大陆需要，通过 HF_ENDPOINT 环境变量配置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Embedding 配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "knowledge_base"

# 检索配置
RETRIEVER_K = 15
RETRIEVER_FETCH_K = 20
RETRIEVER_LAMBDA_MULT = 0.7

# LLM 配置
LLM_MODEL = "qwen-plus"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT = 120.0

# 阿里云百炼（仅从环境变量读取，绝不硬编码）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# API 认证配置（可选，设置后所有请求需通过 X-API-Key 头验证）
API_AUTH_KEY = os.getenv("API_AUTH_KEY")

# 速率限制（每 IP 每分钟最大请求数）
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# 混合检索配置
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
HYBRID_WEIGHT_VECTOR = float(os.getenv("HYBRID_WEIGHT_VECTOR", "0.5"))
HYBRID_WEIGHT_BM25 = float(os.getenv("HYBRID_WEIGHT_BM25", "0.5"))

# 重排序配置
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"  # CPU太慢，默认关闭；开启需export RERANKER_ENABLED=true
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_K = 5

# Redis 配置（如果不设置 REDIS_HOST 则使用内存存储）
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_TTL = int(os.getenv("REDIS_TTL", "86400"))  # 默认 24 小时过期

# 对话历史配置
MAX_HISTORY_LENGTH = 3

# Guardrail 配置
GUARDRAIL_ENABLED = os.getenv("GUARDRAIL_ENABLED", "true").lower() == "true"
DEFAULT_USER_ROLE = os.getenv("DEFAULT_USER_ROLE", "internal")

# 服务配置
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", "8000"))
API_VERSION = "2.1.0"
