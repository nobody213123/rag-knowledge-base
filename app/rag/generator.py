"""
生成模块
负责 LLM 调用和 Prompt 构建

核心设计：
- 使用 AsyncOpenAI 异步客户端，避免阻塞 FastAPI 事件循环
- 支持自动重试（限流/超时/服务端错误各 3 次）
- 所有失败路径返回友好提示，不会抛未处理异常
"""
import asyncio
import time
from openai import AsyncOpenAI
from openai import APIError, APITimeoutError, RateLimitError
from app.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)
from app.logger import get_logger

logger = get_logger("generator")

_client = None  # AsyncOpenAI 客户端（单例）

# 重试策略
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # 基础退避时间（秒）

SYSTEM_PROMPT = """你是专业知识库助手。严格遵守以下规则：

【核心规则】
1. 只能使用"参考资料"中的内容回答，禁止使用任何外部知识
2. 如果参考资料中没有与问题相关的内容，必须回答"暂无相关信息，请联系人工客服"
3. 判断"相关"的标准：问题的主题必须与参考资料中某段内容直接对应
4. 不要推测、联想、引申，只回答参考资料中明确提到的内容

【溯源规则】
- 如果引用了某段参考资料，在引用处加上 [数字] 标记
- 例如：保修期为一年[1]，退货政策为7天无理由[2]

【回答格式】
- 回答简洁通顺，用中文
- 如果有相关信息，直接给出答案
- 如果没有相关信息，只说"暂无相关信息，请联系人工客服"，不要解释原因

【参考资料】
{context}"""


def get_client() -> AsyncOpenAI:
    """获取 LLM 客户端（单例），返回 AsyncOpenAI 实例"""
    global _client
    if _client is None:
        logger.info("正在初始化 LLM 客户端（异步模式）...")
        _client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=LLM_TIMEOUT,
        )
        logger.info("LLM 客户端初始化完成")
    return _client


def build_messages(
    context: str,
    question: str,
    history_context: str = "",
) -> list[dict]:
    """
    构建发送给 LLM 的消息列表
    - system: 系统指令 + 参考资料
    - user: 历史对话（如有） + 当前问题
    """
    sys_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": sys_prompt}]

    if history_context:
        messages.append({
            "role": "user",
            "content": f"历史对话：\n{history_context}\n\n当前问题：{question}",
        })
    else:
        messages.append({"role": "user", "content": question})

    return messages


async def generate(messages: list[dict]) -> tuple[str, float]:
    """
    调用 LLM 生成回答（异步）
    使用 AsyncOpenAI 客户端，不阻塞事件循环
    返回 (answer, cost_ms)，失败时返回友好错误提示
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            llm_start = time.time()
            logger.info(f"正在调用 LLM (尝试 {attempt}/{MAX_RETRIES})...")

            # 异步 API 调用，不阻塞事件循环
            response = await get_client().chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )

            cost_ms = round((time.time() - llm_start) * 1000, 2)
            answer = response.choices[0].message.content
            if answer is None:
                raise ValueError("LLM 返回了空的回答")

            logger.info(f"LLM 回答完成: 耗时 {cost_ms}ms")
            return answer, cost_ms

        except RateLimitError as e:
            # 限流：递增退避时间
            last_error = e
            logger.warning(f"LLM 限流 (尝试 {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * attempt)

        except APITimeoutError as e:
            # 超时：固定间隔重试
            last_error = e
            logger.warning(f"LLM 超时 (尝试 {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

        except APIError as e:
            # API 错误（如 500）：重试
            last_error = e
            logger.error(f"LLM API 错误 (尝试 {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

        except Exception as e:
            # 兜底：所有未分类异常
            last_error = e
            logger.error(f"LLM 未知错误 (尝试 {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

    # 所有重试都失败，返回友好提示而非抛出异常
    error_msg = f"LLM 调用失败（已重试 {MAX_RETRIES} 次）: {last_error}"
    logger.error(error_msg)
    return "抱歉，AI 服务暂时不可用，请稍后重试。", 0.0
