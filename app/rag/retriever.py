"""
检索模块
负责 Embedding 模型、向量库、检索器的初始化与调用

检索链路（按执行顺序）：
1. 向量检索（语义相似度）
2. BM25 检索（关键词匹配）
3. RRF 融合（加权合并两路结果）
4. CrossEncoder 重排序（精排 Top-K）
"""
import re
import threading
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from app.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA_MULT,
    HYBRID_SEARCH_ENABLED,
    HYBRID_WEIGHT_VECTOR,
    HYBRID_WEIGHT_BM25,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_TOP_K,
)
from app.logger import get_logger

logger = get_logger("retriever")

# 单例模式：模块级缓存，避免重复加载模型
_embeddings = None
_vector_store = None
_retriever = None
_bm25 = None
_bm25_corpus = None
_bm25_metadatas = None
_reranker = None
_bm25_lock = threading.Lock()


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
    """获取向量库（单例），从磁盘加载已持久化的 ChromaDB"""
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
    """获取向量检索器（单例），使用 MMR 算法平衡相似度与多样性"""
    global _retriever
    if _retriever is None:
        logger.info("正在配置检索器（MMR）...")
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


# ============================================================
# BM25 关键词检索
# ============================================================

def _chinese_tokenize(text: str) -> list[str]:
    """
    中文分词器（BM25 用）
    先按标点/空格切分，再对中文段做二元分词（bigram）
    无需依赖 jieba，轻量可用
    """
    tokens = []
    pattern = "[\\s,，。！？、；：\u201c\u201d\u2018\u2019\uff08\uff09()\\[\\]{}]+"
    for segment in re.split(pattern, text):
        if not segment:
            continue
        if re.search(r'[\u4e00-\u9fff]', segment):
            # 中文段：二元分词（如"保修期" → ["保修", "修期"]）
            for i in range(len(segment) - 1):
                tokens.append(segment[i:i+2])
            tokens.append(segment)  # 保留整词
        else:
            # 非中文段：原样保留
            tokens.append(segment.lower())
    return tokens


def _init_bm25():
    """从 ChromaDB 读取全量文档构建 BM25 索引（单例，懒加载 + 并发锁）"""
    global _bm25, _bm25_corpus, _bm25_metadatas
    if _bm25 is not None:
        return

    with _bm25_lock:
        # 二次检查：持有锁后再次确认未被其他线程初始化
        if _bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 未安装，BM25 混合检索不可用")
            _bm25 = False
            return

        logger.info("正在从 ChromaDB 读取文档构建 BM25 索引...")
        vector_store = get_vector_store()
        all_data = vector_store.get()

        if not all_data or not all_data.get("documents"):
            logger.warning("ChromaDB 中无文档，BM25 索引为空")
            _bm25 = False
            return

        _bm25_corpus = all_data["documents"]
        _bm25_metadatas = all_data.get("metadatas", [{}] * len(_bm25_corpus))
        tokenized = [_chinese_tokenize(doc) for doc in _bm25_corpus]
        _bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 索引构建完成，共 {len(_bm25_corpus)} 个文档")


def _bm25_search(query: str, k: int) -> list[Document]:
    """执行 BM25 关键词检索，返回 Document 列表"""
    _init_bm25()
    if not _bm25 or not _bm25_corpus:
        return []

    tokenized_query = _chinese_tokenize(query)
    scores = _bm25.get_scores(tokenized_query)
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        metadata = _bm25_metadatas[idx] if idx < len(_bm25_metadatas) else {}
        results.append(Document(
            page_content=_bm25_corpus[idx],
            metadata=metadata,
        ))
    return results


# ============================================================
# CrossEncoder 重排序
# ============================================================

def _rerank_docs(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """
    使用 CrossEncoder 对文档进行重排序
    输入：检索结果文档列表
    输出：按相关性得分降序排列的文档列表（最多 top_k 条）
    模型加载失败时原样返回
    """
    global _reranker
    if not docs:
        return docs

    try:
        if _reranker is None:
            logger.info(f"正在加载重排序模型: {RERANKER_MODEL}...")
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL)
            logger.info("重排序模型加载完成")

        pairs = [[query, doc.page_content] for doc in docs]
        scores = _reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = [doc for doc, _ in ranked[:top_k]]
        logger.info(f"重排序完成: {len(docs)} → {len(result)} 条")
        return result
    except Exception as e:
        logger.warning(f"重排序失败，使用原始排序: {e}")
        return docs[:top_k]


# ============================================================
# RRF（Reciprocal Rank Fusion）融合
# ============================================================

def _rrf_fusion(
    vector_docs: list[Document],
    bm25_docs: list[Document],
    k: int,
    weight_vector: float,
    weight_bm25: float,
) -> list[Document]:
    """
    用 RRF 算法融合两路检索结果
    每篇文档的融合得分 = Σ weight / (rank + 60)
    60 是 RRF 常数，防止极端排名主导
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + weight_vector / (rank + 60)
        if key not in doc_map:
            doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + weight_bm25 / (rank + 60)
        if key not in doc_map:
            doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:k]]


# ============================================================
# 混合检索入口
# ============================================================

def hybrid_retrieve(query: str, k: int = None) -> list[Document]:
    """
    混合检索：向量语义 + BM25 关键词 → RRF 融合 → CrossEncoder 重排

    参数：
        query: 用户问题
        k: 最终返回的文档数（默认 RETRIEVER_K）

    返回：
        list[Document]: 按相关性降序排列的文档列表
    """
    if k is None:
        k = RETRIEVER_K

    if not HYBRID_SEARCH_ENABLED:
        return get_retriever().invoke(query)

    # 阶段一：向量检索（语义相似度）
    vector_docs = get_retriever().invoke(query)
    logger.info(f"向量检索: {len(vector_docs)} 条")

    # 阶段二：BM25 检索（关键词匹配）
    bm25_fetch_k = max(k * 2, RETRIEVER_FETCH_K)
    bm25_docs = _bm25_search(query, bm25_fetch_k)
    logger.info(f"BM25 检索: {len(bm25_docs)} 条")

    # 阶段三：RRF 融合
    fused = _rrf_fusion(
        vector_docs, bm25_docs, k,
        HYBRID_WEIGHT_VECTOR, HYBRID_WEIGHT_BM25,
    )
    logger.info(f"RRF 融合: {len(fused)} 条")

    # 阶段四：CrossEncoder 重排序
    if RERANKER_ENABLED:
        result = _rerank_docs(query, fused, RERANKER_TOP_K)
    else:
        result = fused

    return result


# ============================================================
# 索引管理
# ============================================================

def build_vector_store(chunks, persist_dir: str = None):
    """构建向量库（用于 scripts/build_index.py 索引构建）"""
    if persist_dir is None:
        persist_dir = str(CHROMA_PERSIST_DIR)

    embeddings = get_embeddings()
    logger.info(f"正在构建向量库，存储到: {persist_dir}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )

    logger.info(f"向量库构建完成，共 {len(chunks)} 个向量")
    return vector_store


def clear_vector_store():
    """清空向量库（用于重建索引前的清理）"""
    global _vector_store, _retriever, _bm25, _bm25_corpus, _bm25_metadatas
    vector_store = get_vector_store()
    vector_store.delete_collection()
    _vector_store = None
    _retriever = None
    _bm25 = None
    _bm25_corpus = None
    _bm25_metadatas = None
    logger.info("向量库已清空")
