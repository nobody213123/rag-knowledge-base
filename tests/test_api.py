"""
API 端点测试
使用 FastAPI TestClient 测试实际路由
"""
from unittest.mock import patch
from fastapi.testclient import TestClient


def make_app():
    """创建测试用 app，避免加载真实 RAG 引擎"""
    from fastapi import FastAPI
    from app.config import API_VERSION

    app = FastAPI(version=API_VERSION)

    from app.api.chat import router as chat_router
    from app.api.system import router as system_router
    app.include_router(chat_router, prefix="/chat", tags=["问答"])
    app.include_router(system_router, prefix="/system", tags=["系统"])
    return app


client = TestClient(make_app())


# ========== 健康检查 ==========

def test_health_check():
    resp = client.get("/system/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "rag_engine_loaded" in data


# ========== 统计接口 ==========

def test_stats_initial():
    resp = client.get("/system/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_queries"] == 0


# ========== 问答接口（mock pipeline）==========

MOCK_RESULT = {
    "answer": "测试答案",
    "retrieved_sources": ["docs/test.txt"],
    "sources_detail": [{"index": 1, "file": "test.txt", "full_path": "docs/test.txt", "preview": "内容"}],
    "retrieve_cost_ms": 10.0,
    "llm_cost_ms": 100.0,
    "total_cost_ms": 110.0,
}


@patch("app.api.chat.pipeline.ask", return_value=MOCK_RESULT)
def test_ask_endpoint(mock_ask):
    resp = client.post("/chat/ask", json={"question": "测试问题"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "测试答案"
    assert data["retrieve_cost_ms"] == 10.0
    mock_ask.assert_called_once_with("测试问题")


@patch("app.api.chat.pipeline.ask", return_value=MOCK_RESULT)
def test_ask_empty_question(mock_ask):
    resp = client.post("/chat/ask", json={"question": ""})
    assert resp.status_code == 400
    assert "不能为空" in resp.json()["detail"]
    mock_ask.assert_not_called()


@patch("app.api.chat.pipeline.ask", return_value=MOCK_RESULT)
def test_ask_whitespace_only(mock_ask):
    resp = client.post("/chat/ask", json={"question": "   "})
    assert resp.status_code == 400
    mock_ask.assert_not_called()


@patch("app.api.chat.pipeline.ask_with_history", return_value={**MOCK_RESULT, "history_length": 1})
def test_chat_endpoint(mock_chat):
    resp = client.post("/chat/chat", json={"question": "对话问题", "use_history": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "测试答案"
    assert data["history_length"] == 1
    mock_chat.assert_called_once_with("对话问题", session_id="default", use_history=True)


@patch("app.api.chat.pipeline.ask_with_history", return_value={**MOCK_RESULT, "history_length": 1})
def test_chat_with_session_id(mock_chat):
    resp = client.post("/chat/chat", json={
        "question": "问题", "session_id": "test-session", "use_history": True,
    })
    assert resp.status_code == 200
    mock_chat.assert_called_once_with("问题", session_id="test-session", use_history=True)


@patch("app.api.chat.pipeline.ask_with_history", return_value={**MOCK_RESULT, "history_length": 1})
def test_chat_empty_question(mock_chat):
    resp = client.post("/chat/chat", json={"question": "", "use_history": True})
    assert resp.status_code == 400
    mock_chat.assert_not_called()


# ========== 历史记录 ==========

def test_get_history_default():
    resp = client.get("/chat/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["history"] == []


def test_clear_history():
    resp = client.post("/chat/history/clear")
    assert resp.status_code == 200
    assert "已清空" in resp.json()["message"]
