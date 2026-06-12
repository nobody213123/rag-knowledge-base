"""
API 端点测试
使用 FastAPI TestClient 测试实际路由

注意：pipeline 已改为 async，Mock 需要使用 AsyncMock
"""
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


def make_app():
    """创建测试用 FastAPI 实例（跳过 lifespan 中的 RAG 引擎初始化）"""
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
    """健康检查应返回服务状态和版本信息"""
    resp = client.get("/system/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "rag_engine_loaded" in data


# ========== 统计接口 ==========

def test_stats_initial():
    """初始状态下统计数据应为零"""
    resp = client.get("/system/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_queries"] == 0


# ========== 问答接口（mock 异步 pipeline）==========

MOCK_RESULT = {
    "answer": "测试答案",
    "retrieved_sources": ["docs/test.txt"],
    "sources_detail": [{
        "index": 1, "file": "test.txt", "full_path": "docs/test.txt", "preview": "内容"
    }],
    "retrieve_cost_ms": 10.0,
    "llm_cost_ms": 100.0,
    "total_cost_ms": 110.0,
}


@patch("app.api.chat.pipeline.ask", new_callable=AsyncMock)
def test_ask_endpoint(mock_ask):
    """正常问答应返回答案和耗时数据"""
    mock_ask.return_value = MOCK_RESULT
    resp = client.post("/chat/ask", json={"question": "测试问题"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "测试答案"
    assert data["retrieve_cost_ms"] == 10.0
    mock_ask.assert_called_once_with("测试问题")


@patch("app.api.chat.pipeline.ask", new_callable=AsyncMock)
def test_ask_empty_question(mock_ask):
    """空问题应返回 400"""
    resp = client.post("/chat/ask", json={"question": ""})
    assert resp.status_code == 400
    assert "不能为空" in resp.json()["detail"]
    mock_ask.assert_not_called()


@patch("app.api.chat.pipeline.ask", new_callable=AsyncMock)
def test_ask_whitespace_only(mock_ask):
    """纯空格问题应返回 400"""
    resp = client.post("/chat/ask", json={"question": "   "})
    assert resp.status_code == 400
    mock_ask.assert_not_called()


@patch("app.api.chat.pipeline.ask_with_history", new_callable=AsyncMock)
def test_chat_endpoint(mock_chat):
    """多轮对话应返回答案和历史长度"""
    mock_chat.return_value = {**MOCK_RESULT, "history_length": 1}
    resp = client.post("/chat/chat", json={"question": "对话问题", "use_history": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "测试答案"
    assert data["history_length"] == 1
    mock_chat.assert_called_once_with("对话问题", session_id="default", use_history=True)


@patch("app.api.chat.pipeline.ask_with_history", new_callable=AsyncMock)
def test_chat_with_session_id(mock_chat):
    """支持自定义 session_id"""
    mock_chat.return_value = {**MOCK_RESULT, "history_length": 1}
    resp = client.post("/chat/chat", json={
        "question": "问题", "session_id": "test-session", "use_history": True,
    })
    assert resp.status_code == 200
    mock_chat.assert_called_once_with("问题", session_id="test-session", use_history=True)


@patch("app.api.chat.pipeline.ask_with_history", new_callable=AsyncMock)
def test_chat_empty_question(mock_chat):
    """空问题多轮对话应返回 400"""
    resp = client.post("/chat/chat", json={"question": "", "use_history": True})
    assert resp.status_code == 400
    mock_chat.assert_not_called()


# ========== 历史记录 ==========

def test_get_history_default():
    """默认会话的历史应为空"""
    resp = client.get("/chat/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["history"] == []


def test_get_history_with_session():
    """指定不存在的 session 应返回空列表"""
    resp = client.get("/chat/history?session_id=test-session")
    assert resp.status_code == 200
    data = resp.json()
    assert data["history"] == []


def test_clear_history():
    """清空历史应返回成功消息"""
    resp = client.post("/chat/history/clear")
    assert resp.status_code == 200
    assert "已清空" in resp.json()["message"]
