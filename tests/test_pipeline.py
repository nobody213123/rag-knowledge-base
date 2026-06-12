"""
Pipeline 单元测试
测试 format_docs_with_source、session 历史管理等纯函数
"""
from app.rag.pipeline import format_docs_with_source, get_history, clear_history


class FakeDoc:
    def __init__(self, content, source=""):
        self.page_content = content
        self.metadata = {"source": source}


# ========== format_docs_with_source 测试 ==========

def test_format_docs_single():
    docs = [FakeDoc("这是第一个文档", "docs/a.txt")]
    context, sources = format_docs_with_source(docs)
    assert context == "[1] 这是第一个文档"
    assert len(sources) == 1
    assert sources[0]["file"] == "a.txt"


def test_format_docs_multiple():
    docs = [
        FakeDoc("文档A", "docs/a.txt"),
        FakeDoc("文档B", "docs/b.txt"),
        FakeDoc("文档C", "docs/c.txt"),
    ]
    context, sources = format_docs_with_source(docs)
    assert "[1] 文档A" in context
    assert "[2] 文档B" in context
    assert "[3] 文档C" in context
    assert len(sources) == 3


def test_format_docs_empty():
    context, sources = format_docs_with_source([])
    assert context == ""
    assert sources == []


def test_format_docs_preview_truncated():
    long_content = "A" * 200
    docs = [FakeDoc(long_content, "docs/long.txt")]
    context, sources = format_docs_with_source(docs)
    assert sources[0]["preview"].endswith("...")
    assert len(sources[0]["preview"]) == 103  # 100 + "..."


def test_format_docs_source_without_slash():
    docs = [FakeDoc("内容", "readme.txt")]
    context, sources = format_docs_with_source(docs)
    assert sources[0]["file"] == "readme.txt"


# ========== session 历史管理测试 ==========

def test_get_history_default_empty():
    assert get_history() == []


def test_get_history_nonexistent_session():
    assert get_history("nonexistent") == []


def test_clear_history_nonexistent_session():
    clear_history("no_such_session")
    assert True  # should not raise


def test_get_history_isolated_sessions():
    assert get_history("session_a") == []
    assert get_history("session_b") == []
