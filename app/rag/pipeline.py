"""
RAG Pipeline
串联检索、格式化、生成的完整流程
"""
import time
from collections import defaultdict
from app.config import MAX_HISTORY_LENGTH
from app.logger import get_logger
from app.rag.retriever import get_retriever
from app.rag.generator import build_messages, generate

logger = get_logger("pipeline")

# 基于 session_id 的对话历史管理
conversation_histories: dict[str, list[dict]] = defaultdict(list)


def format_docs_with_source(docs) -> tuple[str, list[dict]]:
    """格式化文档，保留溯源信息"""
    formatted = []
    sources_detail = []

    for i, doc in enumerate(docs):
        formatted.append(f"[{i+1}] {doc.page_content}")
        source = doc.metadata.get("source", "")
        sources_detail.append({
            "index": i + 1,
            "file": source.split("/")[-1] if "/" in source else source,
            "full_path": source,
            "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
        })

    return "\n\n".join(formatted), sources_detail


def ask(question: str, history_context: str = "") -> dict:
    """单次问答：返回答案、检索文档、各阶段耗时"""
    total_start = time.time()
    logger.info(f"收到问题: {question[:50]}...")

    # 检索
    retrieve_start = time.time()
    docs = get_retriever().invoke(question)
    retrieve_cost = round((time.time() - retrieve_start) * 1000, 2)
    logger.info(f"检索完成: {len(docs)} 条结果, 耗时 {retrieve_cost}ms")

    retrieved_sources = [doc.metadata.get("source", "") for doc in docs]
    context, sources_detail = format_docs_with_source(docs)

    # 生成
    messages = build_messages(context, question, history_context)
    answer, llm_cost = generate(messages)
    total_cost = round((time.time() - total_start) * 1000, 2)

    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": retrieved_sources,
        "sources_detail": sources_detail,
        "retrieve_cost_ms": retrieve_cost,
        "llm_cost_ms": llm_cost,
        "total_cost_ms": total_cost,
    }


def ask_with_history(question: str, session_id: str = "default", use_history: bool = True) -> dict:
    """支持多轮对话的问答（按 session_id 隔离历史）"""
    history = conversation_histories[session_id]

    # 构建历史上下文
    history_context = ""
    if use_history and history:
        history_lines = []
        for h in history[-MAX_HISTORY_LENGTH:]:
            history_lines.append(f"用户：{h['question']}")
            history_lines.append(f"助手：{h['answer']}")
        history_context = "\n".join(history_lines)

    # 调用单次问答
    result = ask(question, history_context=history_context)

    # 保存到历史
    history.append({
        "question": question,
        "answer": result["answer"],
    })

    # 历史过长时截断
    if len(history) > MAX_HISTORY_LENGTH * 2:
        conversation_histories[session_id] = history[-MAX_HISTORY_LENGTH:]

    result["history_length"] = len(conversation_histories[session_id])
    return result


def clear_history(session_id: str = "default"):
    """清空指定会话的对话历史"""
    if session_id in conversation_histories:
        conversation_histories[session_id].clear()
        logger.info(f"会话 {session_id} 的历史已清空")
    else:
        logger.info(f"会话 {session_id} 不存在历史记录")


def get_history(session_id: str = "default"):
    """获取指定会话的对话历史"""
    return conversation_histories.get(session_id, [])