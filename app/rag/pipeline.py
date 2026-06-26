"""
编排层：RAG Pipeline 状态机

State: RETRIEVE → JUDGE → GENERATE → VERIFY

设计要点：
- 面向接口编程：pipeline 不直接依赖 retriever/generator，而通过 Tool/Model 抽象
- Judge 阶段判断检索质量，决定是否需要 Query Rewrite
- 全链路 Trace 埋点，每阶段耗时可观测
"""
import asyncio
import re
import time
from enum import Enum
from app.logger import get_logger
from app.rag.retriever import hybrid_retrieve
from app.rag.generator import build_messages
from app.model.registry import get_model_registry
from app.observability.tracer import get_tracer, Trace
from app.observability.metrics import get_collector
from app.memory.store import WindowedMemory
from app.config import MAX_HISTORY_LENGTH, GUARDRAIL_ENABLED, DEFAULT_USER_ROLE, RETRIEVER_K

logger = get_logger("pipeline")

_memory = WindowedMemory(max_turns=MAX_HISTORY_LENGTH)


# ============================================================
# 代词消解：多轮对话中"它/那它" → 上一轮问题中的实体
# ============================================================

_RE_KNOWN = re.compile(
    r'^(P20 Pro|Watch S3|FreeBuds Pro 3|T10|'
    r'Power \d+(?: Pro)?|Lingxi|SoundBox Mini)'
)
_RE_ENTITY = re.compile(r'^(员工|公司|试用期)')
_RE_DE = re.compile(r'^(.+?)的')
_RE_Q = re.compile(
    r'^(.+?)(?:多少钱|几多钱|吗|如何|呢|什么|'
    r'支持|能|会|有|是|打|适合|保修|可以)'
)
_RE_NOUN = re.compile(r'(?:买|选|推荐|考虑)(?:个|款|只|副|了|一个|一款|一副|一只)?([^，。？、\s]{2,10})(?:，|。|$)')
_RE_IT = re.compile(r'(?:那它|它们|它)')


def _extract_entity(q: str) -> str | None:
    for pat in (_RE_KNOWN, _RE_ENTITY):
        m = pat.search(q)
        if m:
            return m.group(1)
    for pat in (_RE_DE, _RE_Q):
        m = pat.match(q)
        if m:
            return m.group(1).strip()
    m = _RE_NOUN.search(q)
    if m:
        return m.group(1).strip()
    return None


def _resolve_pronouns(question: str, history_context: str) -> str:
    if not history_context or not _RE_IT.search(question):
        return question
    last_q = ""
    for line in history_context.split('\n'):
        if line.startswith('用户：'):
            last_q = line[3:]
    if not last_q:
        return question
    entity = _extract_entity(last_q)
    if not entity:
        return question
    return _RE_IT.sub(entity, question)


class Stage(Enum):
    INPUT_GUARDRAIL = "input_guardrail"
    RETRIEVE = "retrieve"
    JUDGE = "judge"
    REWRITE = "rewrite"
    GENERATE = "generate"
    OUTPUT_GUARDRAIL = "output_guardrail"
    VERIFY = "verify"


# ============================================================
# 文档格式化（纯函数，保持原接口）
# ============================================================

def format_docs_with_source(docs) -> tuple[str, list[dict]]:
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


# ============================================================
# Judge 阶段 — 判断检索结果是否充分
# ============================================================

_REFUSE_KEYWORDS = ["暂无相关信息", "没有相关", "无相关"]


async def _judge_retrieval(query: str, docs: list, token_acc: list | None = None) -> tuple[bool, str]:
    """
    判断检索内容是否足够回答用户问题
    返回 (is_answerable, reason)
    """
    if not docs:
        return False, "参考资料为空，无法回答"

    context, _ = format_docs_with_source(docs)
    messages = [
        {"role": "system", "content": "你是问答可行性评估助手。判断仅凭提供的参考资料，能否完整准确回答用户的问题。只回答是或否。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}\n\n仅凭参考资料中的内容，能否完整准确回答这个问题？"},
    ]

    model = get_model_registry()
    answer, record = await model.generate("judge", messages)
    if token_acc is not None:
        token_acc.append(record.tokens)

    is_answerable = "是" in answer[:10]
    if not is_answerable:
        return False, "Judge 判定参考资料不足以回答问题"
    return True, ""


# ============================================================
# Verify 阶段 — 检查回答是否忠实于检索结果
# ============================================================

async def _verify_answer(query: str, answer: str, docs: list, token_acc: list | None = None) -> tuple[bool, str]:
    """检查 LLM 回答是否基于检索内容，有无幻觉"""
    context, _ = format_docs_with_source(docs)
    messages = [
        {"role": "system", "content": "判断以下回答是否基于参考资料中的信息。只回答通过或不通过。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}\n\n回答：{answer}\n\n该回答是否忠实于参考资料？"},
    ]

    model = get_model_registry()
    verdict, record = await model.generate("judge", messages)
    if token_acc is not None:
        token_acc.append(record.tokens)

    if "不通过" in verdict:
        is_refuse = any(kw in answer for kw in _REFUSE_KEYWORDS)
        if is_refuse:
            return True, "拒答无需验证"
        return False, "Verify 判定回答可能超出检索范围"

    return True, ""


# ============================================================
# 主流程
# ============================================================

async def run_pipeline(
    question: str,
    history_context: str = "",
    user_role: str | None = None,
) -> dict:
    """
    状态机执行 RAG 流程（含 Guardrail）

    State 流转:
      INPUT_GUARDRAIL → RETRIEVE → JUDGE
        ┌─ 充分 → GENERATE → OUTPUT_GUARDRAIL → VERIFY
        │          ┌─ 通过 → 返回结果
        │          └─ 不通过 → 返回结果（降级）
        └─ 不充分 → REWRITE → RETRIEVE(2) → GENERATE → OUTPUT_GUARDRAIL → VERIFY

    返回 dict（保持与旧版兼容）:
      {question, answer, retrieved_sources, sources_detail, ...cost_ms}
    """
    if user_role is None:
        user_role = DEFAULT_USER_ROLE if GUARDRAIL_ENABLED else None

    trace: Trace = get_tracer().start(question)
    logger.info(f"Pipeline 启动: {question[:50]}... (role={user_role})")

    total_start = time.time()
    token_acc: list[int] = []

    # ---- Stage 0: INPUT GUARDRAIL ----
    if GUARDRAIL_ENABLED:
        s = trace.stage(Stage.INPUT_GUARDRAIL.value)
        s.start()
        from app.guardrails.input import check_input
        is_safe, reason = await check_input(question)
        s.stop(detail=reason or "通过")

        if not is_safe:
            total_cost = round((time.time() - total_start) * 1000, 2)
            logger.warning(f"Input Guardrail 拦截: {reason}")
            return {
                "question": question,
                "answer": "抱歉，您的输入包含安全限制内容，无法回答。",
                "retrieved_sources": [],
                "sources_detail": [],
                "retrieve_cost_ms": 0,
                "llm_cost_ms": 0,
                "total_cost_ms": total_cost,
                "trace_id": trace.trace_id,
                "total_tokens": sum(token_acc),
            }

    # ---- 代词消解：用历史替换"它/那它"为本轮实体 ----
    retrieval_query = _resolve_pronouns(question, history_context)
    if retrieval_query != question:
        logger.info(f"代词消解: [{question}] → [{retrieval_query}]")

    # ---- Stage 1: RETRIEVE ----
    s = trace.stage(Stage.RETRIEVE.value)
    s.start()
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(None, hybrid_retrieve, retrieval_query, RETRIEVER_K, user_role)
    s.stop(detail=f"{len(docs)} 条结果")

    retrieve_cost = s.cost_ms
    retrieved_sources = [d.metadata.get("source", "") for d in docs]
    context, sources_detail = format_docs_with_source(docs)

    # ---- Stage 2: JUDGE ----
    s = trace.stage(Stage.JUDGE.value)
    s.start()
    is_answerable, reason = await _judge_retrieval(retrieval_query, docs, token_acc)
    s.stop(detail=reason)

    # ---- 可选 Stage: REWRITE + RETRIEVE(2) + RE-JUDGE ----
    if not is_answerable:
        logger.info(f"Judge 判定不可回答，尝试重写查询: {reason}")
        s = trace.stage(Stage.REWRITE.value)
        s.start()
        rewritten = await _rewrite_query(retrieval_query, token_acc)
        s.stop(detail=rewritten)

        s = trace.stage(Stage.RETRIEVE.value)
        s.start()
        docs2 = await loop.run_in_executor(None, hybrid_retrieve, rewritten, RETRIEVER_K, user_role)  # noqa: E501
        s.stop(detail=f"{len(docs2)} 条结果 (重查)")

        if docs2:
            s = trace.stage(Stage.JUDGE.value)
            s.start()
            is_answerable, reason = await _judge_retrieval(rewritten, docs2, token_acc)
            s.stop(detail=f"重判: {reason}")

        if not is_answerable:
            total_cost = round((time.time() - total_start) * 1000, 2)
            logger.info(f"重写后仍不可回答，直接拒答: {reason}")
            return {
                "question": question,
                "answer": "暂无相关信息，请联系人工客服",
                "retrieved_sources": [d.metadata.get("source", "") for d in docs],
                "sources_detail": sources_detail,
                "retrieve_cost_ms": round(retrieve_cost, 2),
                "llm_cost_ms": 0,
                "total_cost_ms": total_cost,
                "trace_id": trace.trace_id,
                "total_tokens": sum(token_acc),
            }

        docs = docs2
        context, sources_detail = format_docs_with_source(docs)
        retrieve_cost += s.cost_ms

    # ---- Stage 3: GENERATE ----
    s = trace.stage(Stage.GENERATE.value)
    s.start()
    messages = build_messages(context, question, history_context)
    model = get_model_registry()
    answer, record = await model.generate("qa", messages)
    token_acc.append(record.tokens)
    s.stop(detail=f"model={record.model}, tokens={record.tokens}")

    llm_cost = s.cost_ms

    # ---- Stage 4: OUTPUT GUARDRAIL ----
    if GUARDRAIL_ENABLED:
        s = trace.stage(Stage.OUTPUT_GUARDRAIL.value)
        s.start()
        from app.guardrails.output import check_output
        clean_answer, has_pii = await check_output(answer)
        s.stop(detail=f"PII={'有' if has_pii else '无'}")
        answer = clean_answer

    # ---- Stage 5: VERIFY ----
    s = trace.stage(Stage.VERIFY.value)
    s.start()
    verified, v_reason = await _verify_answer(question, answer, docs, token_acc)
    s.stop(detail=v_reason)

    if not verified:
        logger.warning(f"Verify 不通过: {v_reason}")
        if any(kw in answer for kw in _REFUSE_KEYWORDS):
            pass
        else:
            answer += "\n\n（注：以上回答可能存在不准确信息，建议人工核实）"

    total_cost = round((time.time() - total_start) * 1000, 2)
    total_tokens = sum(token_acc)

    result = {
        "question": question,
        "answer": answer,
        "retrieved_sources": retrieved_sources,
        "sources_detail": sources_detail,
        "retrieve_cost_ms": round(retrieve_cost, 2),
        "llm_cost_ms": round(llm_cost, 2),
        "total_cost_ms": total_cost,
        "trace_id": trace.trace_id,
        "total_tokens": total_tokens,
    }

    trace.log()
    get_collector().record(result)

    return result


async def _rewrite_query(original: str, token_acc: list | None = None) -> str:
    """查询重写：扩展/改写原始问题"""
    messages = [
        {"role": "system", "content": "改写以下问题，使其更适合知识库检索。保留原意，补充关键术语。直接输出改写结果。"},
        {"role": "user", "content": original},
    ]
    model = get_model_registry()
    rewritten, record = await model.generate("judge", messages)
    if token_acc is not None:
        token_acc.append(record.tokens)
    return rewritten.strip() or original


# ============================================================
# 对外接口（保持签名兼容）
# ============================================================

async def ask(question: str, history_context: str = "") -> dict:
    """单次问答"""
    result = await run_pipeline(question, history_context=history_context)
    return result


async def ask_with_history(
    question: str,
    session_id: str = "default",
    use_history: bool = True,
) -> dict:
    """多轮对话"""
    history = []
    if use_history:
        history = await _memory.get(session_id)
    history_context = _memory.build_context(history) if history else ""

    result = await run_pipeline(question, history_context=history_context)

    await _memory.append(session_id, {
        "question": question,
        "answer": result["answer"],
    })

    updated = await _memory.get(session_id)
    result["history_length"] = len(updated)

    return result


async def clear_history(session_id: str = "default"):
    await _memory.clear(session_id)
    logger.info(f"会话 {session_id} 历史已清空")


async def get_history(session_id: str = "default"):
    return await _memory.get(session_id)
