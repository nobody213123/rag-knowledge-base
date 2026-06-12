"""
问答 API
处理单次问答和多轮对话请求
"""
from fastapi import APIRouter, HTTPException, Query
from app.logger import get_logger
from app.schemas.chat import (
    AskRequest,
    ChatRequest,
    AskResponse,
    ChatResponse,
    HistoryResponse,
)
from app.rag import pipeline
from app.api.system import record_query

logger = get_logger("api.chat")
router = APIRouter()


@router.post("/ask", response_model=AskResponse, summary="单次问答")
async def ask_question(request: AskRequest):
    """单次问答接口（无历史上下文）"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    logger.info(f"收到 API 请求: {request.question[:50]}...")
    result = pipeline.ask(request.question)

    record_query(
        retrieve_ms=result["retrieve_cost_ms"],
        llm_ms=result["llm_cost_ms"],
        total_ms=result["total_cost_ms"],
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["retrieved_sources"],
        sources_detail=result["sources_detail"],
        retrieve_cost_ms=result["retrieve_cost_ms"],
        llm_cost_ms=result["llm_cost_ms"],
        total_cost_ms=result["total_cost_ms"],
    )


@router.post("/chat", response_model=ChatResponse, summary="多轮对话")
async def chat(request: ChatRequest):
    """多轮对话接口（支持历史上下文 + 文档溯源）"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    logger.info(f"收到对话请求 (session={request.session_id}): {request.question[:50]}...")
    result = pipeline.ask_with_history(
        request.question,
        session_id=request.session_id,
        use_history=request.use_history,
    )

    record_query(
        retrieve_ms=result["retrieve_cost_ms"],
        llm_ms=result["llm_cost_ms"],
        total_ms=result["total_cost_ms"],
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["retrieved_sources"],
        sources_detail=result["sources_detail"],
        history_length=result["history_length"],
        retrieve_cost_ms=result["retrieve_cost_ms"],
        llm_cost_ms=result["llm_cost_ms"],
        total_cost_ms=result["total_cost_ms"],
    )


@router.get("/history", response_model=HistoryResponse, summary="获取对话历史")
async def get_history(session_id: str = Query("default", description="会话 ID")):
    """获取指定会话的对话历史"""
    history = pipeline.get_history(session_id)
    return HistoryResponse(history=history, count=len(history))


@router.post("/history/clear", summary="清空对话历史")
async def clear_history(session_id: str = Query("default", description="会话 ID")):
    """清空指定会话的对话历史"""
    pipeline.clear_history(session_id)
    return {"message": f"会话 {session_id} 的历史已清空"}
