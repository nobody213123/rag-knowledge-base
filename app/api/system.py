"""
系统 API
健康检查、调用统计、重建索引

设计要点：
- query_stats 使用 threading.Lock 保护，支持并发写入
- rag_engine_loaded 状态在 lifespan 中设置
"""
import threading
from fastapi import APIRouter, HTTPException
from app.logger import get_logger
from app.config import API_VERSION
from app.schemas.chat import HealthResponse, StatsResponse

logger = get_logger("api.system")
router = APIRouter()

# 调用统计（内存级，服务重启后重置）
# 使用 threading.Lock 保证并发安全
_stats_lock = threading.Lock()
query_stats = {
    "total": 0,
    "retrieve_sum": 0.0,
    "llm_sum": 0.0,
    "total_sum": 0.0,
}

# RAG 引擎加载状态（在 lifespan 中设置）
rag_engine_loaded = False


def set_rag_loaded(loaded: bool):
    """设置 RAG 引擎加载状态（由 main.py lifespan 调用）"""
    global rag_engine_loaded
    rag_engine_loaded = loaded


def record_query(retrieve_ms: float, llm_ms: float, total_ms: float):
    """记录一次查询统计（线程安全）"""
    with _stats_lock:
        query_stats["total"] += 1
        query_stats["retrieve_sum"] += retrieve_ms
        query_stats["llm_sum"] += llm_ms
        query_stats["total_sum"] += total_ms


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """
    健康检查端点（Docker healthcheck 使用）
    返回服务状态、版本号、RAG 引擎加载状态
    """
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        rag_engine_loaded=rag_engine_loaded,
    )


@router.get("/stats", response_model=StatsResponse, summary="调用统计")
async def get_stats():
    """
    获取系统调用统计
    包括总查询数、平均检索耗时、平均 LLM 耗时、平均总耗时
    """
    with _stats_lock:
        total = query_stats["total"]
        if total == 0:
            return StatsResponse(
                total_queries=0, avg_retrieve_ms=0, avg_llm_ms=0, avg_total_ms=0
            )
        return StatsResponse(
            total_queries=total,
            avg_retrieve_ms=round(query_stats["retrieve_sum"] / total, 2),
            avg_llm_ms=round(query_stats["llm_sum"] / total, 2),
            avg_total_ms=round(query_stats["total_sum"] / total, 2),
        )


@router.post("/rebuild", summary="重建知识库索引")
async def rebuild_index():
    """重建知识库索引（异步执行：先用线程池处理同步操作）"""
    logger.info("收到重建索引请求")
    try:
        from app.rag.loader import load_documents, split_documents
        from app.rag.retriever import build_vector_store

        # 加载文档（同步 I/O，通过线程池执行）
        documents = load_documents()
        if not documents:
            raise HTTPException(status_code=400, detail="未找到文档")

        # 分块
        chunks = split_documents(documents)

        # 构建向量库
        build_vector_store(chunks)

        return {"message": "索引重建完成"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重建索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"重建失败: {str(e)}")
