"""
观测层：调用链追踪
每次查询生成 trace_id，记录各阶段耗时
"""
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from app.logger import get_logger

logger = get_logger("tracer")

# 最多保留最近 1000 条 trace，防止内存泄漏
_MAX_TRACES = 1000


@dataclass
class StageRecord:
    name: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    cost_ms: float = 0.0
    detail: str = ""

    def start(self):
        self.start_ms = time.time() * 1000

    def stop(self, detail: str = ""):
        self.end_ms = time.time() * 1000
        self.cost_ms = round(self.end_ms - self.start_ms, 2)
        self.detail = detail


@dataclass
class Trace:
    trace_id: str
    query: str = ""
    stages: dict[str, StageRecord] = field(default_factory=dict)
    error: str = ""

    def stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]

    def summary(self) -> str:
        parts = [f"[{self.trace_id[:8]}] {self.query[:30]}"]
        for name, s in self.stages.items():
            parts.append(f"  {name}: {s.cost_ms}ms")
        return "\n".join(parts)

    def log(self):
        for name, s in self.stages.items():
            logger.info(f"[{self.trace_id[:8]}] {name}: {s.cost_ms}ms | {s.detail}")
        if self.error:
            logger.error(f"[{self.trace_id[:8]}] ERROR: {self.error}")


class Tracer:
    def __init__(self):
        self._traces: deque[Trace] = deque(maxlen=_MAX_TRACES)

    def start(self, query: str) -> Trace:
        t = Trace(trace_id=uuid.uuid4().hex, query=query)
        self._traces.append(t)
        return t

    def get_recent(self, n: int = 10) -> list[Trace]:
        return list(self._traces)[-n:]


_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer
