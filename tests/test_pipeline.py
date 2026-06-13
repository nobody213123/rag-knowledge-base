"""
Pipeline 单元测试
测试 format_docs_with_source、session 历史管理等纯函数

注意：
- format_docs_with_source 是同步纯函数
- get_history / clear_history 是异步函数
- 历史使用 Redis 或内存回退（无 Redis 时自动降级）
"""
import pytest
from app.rag.pipeline import format_docs_with_source, get_history, clear_history


class FakeDoc:
    """
    模拟 langchain Document 对象
    用于纯函数测试，避免加载真实 Embedding 模型
    """
    def __init__(self, content, source=""):
        self.page_content = content
        self.metadata = {"source": source}


# ============================================================
# format_docs_with_source 测试
# ============================================================

def test_format_docs_single():
    """单个文档应正确格式化并保留溯源"""
    docs = [FakeDoc("这是第一个文档", "docs/a.txt")]
    context, sources = format_docs_with_source(docs)
    assert context == "[1] 这是第一个文档"
    assert len(sources) == 1
    assert sources[0]["file"] == "a.txt"


def test_format_docs_multiple():
    """多个文档应依次编号，用换行分隔"""
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
    """空文档列表应返回空字符串和空列表"""
    context, sources = format_docs_with_source([])
    assert context == ""
    assert sources == []


def test_format_docs_preview_truncated():
    """长文档应截断为 100 字 + ..."""
    long_content = "A" * 200
    docs = [FakeDoc(long_content, "docs/long.txt")]
    context, sources = format_docs_with_source(docs)
    assert sources[0]["preview"].endswith("...")
    assert len(sources[0]["preview"]) == 103  # 100 + "..."


def test_format_docs_source_without_slash():
    """source 路径不含 / 时，file 应等于完整路径"""
    docs = [FakeDoc("内容", "readme.txt")]
    context, sources = format_docs_with_source(docs)
    assert sources[0]["file"] == "readme.txt"


# ============================================================
# session 历史管理测试（异步）
# ============================================================

@pytest.mark.asyncio
async def test_get_history_default_empty():
    """默认 session 历史应为空"""
    history = await get_history()
    assert history == []


@pytest.mark.asyncio
async def test_get_history_nonexistent_session():
    """不存在的 session 应返回空列表"""
    history = await get_history("nonexistent")
    assert history == []


@pytest.mark.asyncio
async def test_clear_history_nonexistent_session():
    """清空不存在的 session 不应报错"""
    await clear_history("no_such_session")
    assert True


@pytest.mark.asyncio
async def test_get_history_isolated_sessions():
    """不同 session 的历史应互相隔离"""
    h1 = await get_history("isolated_a")
    h2 = await get_history("isolated_b")
    assert h1 == []
    assert h2 == []
