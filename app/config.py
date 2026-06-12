"""
全局配置模块
所有配置集中管理，避免到处找配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# HuggingFace 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
RETRIEVER_K = 10
RETRIEVER_FETCH_K = 20
RETRIEVER_LAMBDA_MULT = 0.7

# 分块配置
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# LLM 配置
LLM_MODEL = "deepseek-r1-distill-qwen-7b"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT = 90.0

# 阿里云百炼
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

# 对话历史配置
MAX_HISTORY_LENGTH = 3

# 服务配置
API_HOST = "0.0.0.0"
API_PORT = 8000
API_VERSION = "2.0.0"
