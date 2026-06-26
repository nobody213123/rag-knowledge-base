"""
RAG 知识库 Web 服务入口
FastAPI 应用，挂载路由、中间件、前端静态文件

中间件栈（从上到下）：
1. CORS — 允许跨域请求
2. RateLimit — 限制单 IP 请求频率 (app/middleware/rate_limiter.py)
3. Auth — API Key 认证（可选）
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.logger import get_logger
from app.config import API_VERSION, RATE_LIMIT_PER_MINUTE, API_AUTH_KEY
from app.api.chat import router as chat_router
from app.api.system import router as system_router, set_rag_loaded
from app.middleware.rate_limiter import RateLimiter

logger = get_logger("main")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import DASHSCOPE_API_KEY
    if not DASHSCOPE_API_KEY:
        logger.error("环境变量 DASHSCOPE_API_KEY 未设置")
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

    # 预热模型注册表
    try:
        from app.model.registry import get_model_registry
        get_model_registry()
        logger.info("模型注册表初始化完成")
    except Exception as e:
        logger.warning(f"模型注册表初始化失败: {e}")

    yield

    try:
        from app.memory.store import close
        await close()
    except Exception:
        pass
    logger.info("Web 服务关闭")


app = FastAPI(
    title="RAG 知识库问答系统",
    description="基于 RAG 架构的企业级私有知识库智能问答 API",
    version=API_VERSION,
    lifespan=lifespan,
)


# 1. CORS — 通过环境变量 CORS_ORIGINS 配置允许的域名，逗号分隔
import os
_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 速率限制
rate_limiter = RateLimiter(limit=RATE_LIMIT_PER_MINUTE)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith(("/chat", "/system")):
        await rate_limiter.check(request)
    response = await call_next(request)
    return response


# 3. API Key 认证（可选）
if API_AUTH_KEY:

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        public_paths = ["/", "/docs", "/openapi.json"]
        if any(request.url.path == p or request.url.path.startswith(p.rstrip("/"))
               for p in public_paths):
            return await call_next(request)
        api_key = request.headers.get("X-API-Key")
        if api_key != API_AUTH_KEY:
            raise HTTPException(status_code=401, detail="无效的 API Key")
        return await call_next(request)


# 路由
app.include_router(chat_router, prefix="/chat", tags=["问答"])
app.include_router(system_router, prefix="/system", tags=["系统"])


@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "RAG 知识库问答系统", "docs": "/docs"}


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    from app.config import API_HOST, API_PORT
    logger.info(f"启动 Web 服务: http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
