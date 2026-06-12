"""
RAG 知识库 Web 服务
FastAPI 入口，挂载路由 + 前端静态文件
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.logger import get_logger
from app.config import API_VERSION
from app.api.chat import router as chat_router
from app.api.system import router as system_router, set_rag_loaded

logger = get_logger("main")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在初始化 RAG 引擎...")
    from app.rag.retriever import get_retriever
    get_retriever()
    set_rag_loaded(True)
    logger.info("RAG 引擎初始化完成")
    yield
    logger.info("Web 服务关闭")


app = FastAPI(
    title="RAG 知识库问答系统",
    description="基于 RAG 架构的企业级私有知识库智能问答 API",
    version=API_VERSION,
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/chat", tags=["问答"])
app.include_router(system_router, prefix="/system", tags=["系统"])


@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "RAG 知识库问答系统", "docs": "/docs"}


# 挂载前端静态文件（CSS/JS）
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    from app.config import API_HOST, API_PORT

    logger.info(f"启动 Web 服务: http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
