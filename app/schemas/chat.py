"""
数据模型定义
Request / Response 格式
"""
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    question: str
    use_history: bool = True


class SourceItem(BaseModel):
    index: int
    file: str
    full_path: str
    preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    sources_detail: list[SourceItem]
    retrieve_cost_ms: float
    llm_cost_ms: float
    total_cost_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    sources_detail: list[SourceItem]
    history_length: int
    retrieve_cost_ms: float
    llm_cost_ms: float
    total_cost_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    rag_engine_loaded: bool


class StatsResponse(BaseModel):
    total_queries: int
    avg_retrieve_ms: float
    avg_llm_ms: float
    avg_total_ms: float


class HistoryResponse(BaseModel):
    history: list[dict]
    count: int
