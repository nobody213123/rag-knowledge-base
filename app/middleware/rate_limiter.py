"""
入口层：速率限制中间件
简易内存 Token Bucket，每个 IP 每分钟限流
"""
import time
from fastapi import Request, HTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from app.config import RATE_LIMIT_PER_MINUTE

# 每 N 次 check 清理一次过期 IP
_CLEANUP_INTERVAL = 100


class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT_PER_MINUTE):
        self.limit = limit
        self._records: dict[str, list[float]] = {}
        self._check_count = 0

    async def check(self, request: Request):
        if self.limit <= 0:
            return
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        if client_ip in self._records:
            self._records[client_ip] = [
                t for t in self._records[client_ip] if now - t < 60
            ]
        else:
            self._records[client_ip] = []

        if len(self._records[client_ip]) >= self.limit:
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，每分钟最多 {self.limit} 次",
            )

        self._records[client_ip].append(now)

        # 定期清理过期 IP，防止内存泄漏
        self._check_count += 1
        if self._check_count >= _CLEANUP_INTERVAL:
            self._check_count = 0
            empty_ips = [
                ip for ip, times in self._records.items()
                if not times or now - times[-1] > 120
            ]
            for ip in empty_ips:
                del self._records[ip]
