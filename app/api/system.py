"""
系统 API
健康检查、调用统计、重建索引
"""
from fastapi import APIRouter, HTTPException
from app.logger import get_logger
from app.config import API_VERSION
from app.schemas.chat import HealthResponse, StatsResponse

logger = get_logger("api.system")
router = APIRouter()

# 内存统计
query_stats = {
    "total": 0,
    "retrieve_sum": 0.0,
    "llm_sum": 0.0,
    "total_sum": 0.0,
}

rag_engine_loaded = False


def set_rag_loaded(loaded: bool):
    global rag_engine_loaded
    rag_engine_loaded = loaded


def record_query(retrieve_ms: float, llm_ms: float, total_ms: float):
    query_stats["total"] += 1
    query_stats["retrieve_sum"] += retrieve_ms
    query_stats["llm_sum"] += llm_ms
    query_stats["total_sum"] += total_ms


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        rag_engine_loaded=rag_engine_loaded,
    )


@router.get("/stats", response_model=StatsResponse, summary="调用统计")
async def get_stats():
    total = query_stats["total"]
    if total == 0:
        return StatsResponse(total_queries=0, avg_retrieve_ms=0, avg_llm_ms=0, avg_total_ms=0)
    return StatsResponse(
        total_queries=total,
        avg_retrieve_ms=round(query_stats["retrieve_sum"] / total, 2),
        avg_llm_ms=round(query_stats["llm_sum"] / total, 2),
        avg_total_ms=round(query_stats["total_sum"] / total, 2),
    )


@router.post("/rebuild", summary="重建知识库索引")
async def rebuild_index():
    """重建知识库索引"""
    logger.info("收到重建索引请求")
    try:
        from app.rag.loader import load_documents, split_documents
        from app.rag.retriever import build_vector_store

        documents = load_documents()
        if not documents:
            raise HTTPException(status_code=400, detail="未找到文档")
        chunks = split_documents(documents)
        build_vector_store(chunks)
        return {"message": "索引重建完成"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重建索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"重建失败: {str(e)}")
