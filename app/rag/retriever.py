"""
检索模块
负责 Embeddings、向量库、检索器
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from app.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA_MULT,
)
from app.logger import get_logger

logger = get_logger("retriever")

# 延迟初始化，避免 import 时执行
_embeddings = None
_vector_store = None
_retriever = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取 Embedding 模型（单例）"""
    global _embeddings
    if _embeddings is None:
        logger.info(f"正在加载嵌入模型: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> Chroma:
    """获取向量库（单例）"""
    global _vector_store
    if _vector_store is None:
        logger.info("正在加载向量库...")
        _vector_store = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
        logger.info("向量库加载完成")
    return _vector_store


def get_retriever() -> BaseRetriever:
    """获取检索器（单例）"""
    global _retriever
    if _retriever is None:
        _retriever = get_vector_store().as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_LAMBDA_MULT,
            },
        )
        logger.info("检索器配置完成")
    return _retriever


def build_vector_store(chunks, persist_dir: str = None):
    """构建向量库（用于索引构建）"""
    if persist_dir is None:
        persist_dir = str(CHROMA_PERSIST_DIR)

    embeddings = get_embeddings()
    logger.info(f"正在构建向量库，存储到: {persist_dir}")
    print(f"正在构建向量库，存储到：{persist_dir}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )

    logger.info(f"向量库构建完成，共 {len(chunks)} 个向量")
    print(f"向量库构建完成，共 {len(chunks)} 个向量")
    return vector_store


def clear_vector_store():
    """清空向量库"""
    global _vector_store, _retriever
    vector_store = get_vector_store()
    vector_store.delete_collection()
    _vector_store = None
    _retriever = None
    logger.info("向量库已清空")
