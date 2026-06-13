"""
Redis 对话历史存储
支持异步写入，Redis 不可用时自动降级到内存存储

设计要点：
- 使用 redis.asyncio 非阻塞客户端，不阻塞事件循环
- Redis 连接失败时（如未安装 Redis），静默降级到内存字典
- 所有历史自动过期（TTL），避免数据无限堆积
- 内存降级模式也有大小限制，防止内存泄漏
"""
import json
from collections import defaultdict
from app.logger import get_logger
from app.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL, MAX_HISTORY_LENGTH

logger = get_logger("redis_client")

_redis = None
_redis_available = False
_fallback: dict[str, list[dict]] = defaultdict(list)


async def _get_redis():
    """
    获取异步 Redis 客户端（单例 + 懒加载）
    连接失败时将 _redis_available 置为 False，后续请求直接走内存
    """
    global _redis, _redis_available
    if _redis is None and REDIS_HOST:
        try:
            import redis.asyncio as aioredis
            _redis = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await _redis.ping()
            _redis_available = True
            logger.info(f"Redis 连接成功: {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            _redis_available = False
            logger.warning(f"Redis 连接失败，降级到内存存储: {e}")
    elif not REDIS_HOST:
        _redis_available = False
    return _redis if _redis_available else None


async def get_history(session_id: str) -> list[dict]:
    """获取指定会话的对话历史"""
    client = await _get_redis()
    if client:
        try:
            key = f"history:{session_id}"
            items = await client.lrange(key, 0, -1)
            return [json.loads(item) for item in items]
        except Exception as e:
            logger.warning(f"Redis 读取失败，降级到内存: {e}")
    return list(_fallback.get(session_id, []))


async def append_history(session_id: str, entry: dict):
    """
    追加一条对话记录到历史
    Redis：使用 RPUSH + LTRIM 控制长度 + EXPIRE 自动过期
    内存：append 后截断到 MAX_HISTORY_LENGTH * 2，防止内存泄漏
    """
    client = await _get_redis()
    if client:
        try:
            key = f"history:{session_id}"
            await client.rpush(key, json.dumps(entry, ensure_ascii=False))
            # 只保留最近 MAX_HISTORY_LENGTH * 2 条
            await client.ltrim(key, -MAX_HISTORY_LENGTH * 2, -1)
            await client.expire(key, REDIS_TTL)
            return
        except Exception as e:
            logger.warning(f"Redis 写入失败，降级到内存: {e}")
    _fallback[session_id].append(entry)
    # 内存模式也做截断，防止单 session 无限增长
    if len(_fallback[session_id]) > MAX_HISTORY_LENGTH * 2:
        _fallback[session_id] = _fallback[session_id][-(MAX_HISTORY_LENGTH * 2):]


async def clear_history(session_id: str):
    """清空指定会话的对话历史"""
    client = await _get_redis()
    if client:
        try:
            await client.delete(f"history:{session_id}")
            return
        except Exception as e:
            logger.warning(f"Redis 删除失败，降级到内存: {e}")
    _fallback.pop(session_id, None)


async def close():
    """关闭 Redis 连接（应用关闭时调用）"""
    global _redis
    if _redis:
        try:
            await _redis.close()
        except Exception:
            pass
        _redis = None
