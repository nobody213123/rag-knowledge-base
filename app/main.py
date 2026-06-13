"""
RAG 知识库 Web 服务入口
FastAPI 应用，挂载路由、中间件、前端静态文件

中间件栈（从上到下）：
1. CORS — 允许跨域请求
2. RateLimit — 限制单 IP 请求频率
3. Auth — API Key 认证（可选）
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from app.logger import get_logger
from app.config import API_VERSION, RATE_LIMIT_PER_MINUTE, API_AUTH_KEY
from app.api.chat import router as chat_router
from app.api.system import router as system_router, set_rag_loaded

logger = get_logger("main")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ============================================================
# 应用生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时校验配置并初始化 RAG 引擎，关闭时清理资源"""
    # 启动时校验 API Key，避免运行时才崩溃
    from app.config import DASHSCOPE_API_KEY
    if not DASHSCOPE_API_KEY:
        logger.error("环境变量 DASHSCOPE_API_KEY 未设置")
        logger.error("请执行: export DASHSCOPE_API_KEY=sk-xxx")
        set_rag_loaded(False)
        yield
        return

    logger.info("正在初始化 RAG 引擎...")
    try:
        from app.rag.retriever import get_retriever
        get_retriever()
        set_rag_loaded(True)
        logger.info("RAG 引擎初始化完成")
    except Exception as e:
        logger.error(f"RAG 引擎初始化失败: {e}")
        set_rag_loaded(False)
    yield
    # 关闭时清理 Redis 连接
    try:
        from app import redis_client
        await redis_client.close()
    except Exception:
        pass
    logger.info("Web 服务关闭")


app = FastAPI(
    title="RAG 知识库问答系统",
    description="基于 RAG 架构的企业级私有知识库智能问答 API",
    version=API_VERSION,
    lifespan=lifespan,
)


# ============================================================
# 中间件配置
# ============================================================

# 1. CORS — 允许前端跨域访问
#    生产环境应将 "*" 替换为具体的前端域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. 速率限制（基于 IP 的简易内存实现）
class RateLimiter:
    """
    简易内存速率限制器
    限制每个 IP 每分钟的请求数
    生产环境建议替换为 Redis 实现
    """
    def __init__(self, limit: int = RATE_LIMIT_PER_MINUTE):
        self.limit = limit
        self._records: dict[str, list[float]] = {}

    async def check(self, request: Request):
        if self.limit <= 0:
            return
        import time
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # 清理过期记录（60 秒前）
        if client_ip in self._records:
            self._records[client_ip] = [
                t for t in self._records[client_ip] if now - t < 60
            ]
        else:
            self._records[client_ip] = []

        # 检查是否超限
        if len(self._records[client_ip]) >= self.limit:
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，每分钟最多 {self.limit} 次",
            )

        self._records[client_ip].append(now)


rate_limiter = RateLimiter()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """速率限制中间件：对 /chat 和 /system 路由生效"""
    if request.url.path.startswith(("/chat", "/system")):
        await rate_limiter.check(request)
    response = await call_next(request)
    return response


# 3. API Key 认证（可选）
if API_AUTH_KEY:

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        """
        API Key 认证中间件
        客户端需在 Header 中携带 X-API-Key，值须与 API_AUTH_KEY 一致
        前端页面（/）和文档（/docs）跳过认证
        """
        # 允许路径：前端页面、Swagger 文档、健康检查
        public_paths = ["/", "/docs", "/openapi.json"]
        if any(request.url.path == p or request.url.path.startswith(p.rstrip("/"))
               for p in public_paths):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key != API_AUTH_KEY:
            raise HTTPException(status_code=401, detail="无效的 API Key")

        return await call_next(request)


# ============================================================
# 路由注册
# ============================================================
app.include_router(chat_router, prefix="/chat", tags=["问答"])
app.include_router(system_router, prefix="/system", tags=["系统"])


@app.get("/")
async def root():
    """根路径：返回前端页面或 API 信息"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "RAG 知识库问答系统", "docs": "/docs"}


# 挂载前端静态文件（CSS/JS 等）
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ============================================================
# 直接运行入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    from app.config import API_HOST, API_PORT

    logger.info(f"启动 Web 服务: http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
