"""
pytest 全局配置
确保 Redis 回退存储在各测试间隔离
"""
import pytest


@pytest.fixture(autouse=True)
def _reset_redis_fallback():
    """每个测试前清空 Redis 回退的内存存储，保证测试隔离"""
    from app.memory.store import _fallback
    _fallback.clear()
    yield
