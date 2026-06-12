"""
生成器单元测试
测试 app.rag.generator 中的纯函数和异步重试逻辑

注意：
- build_messages 是纯函数，可直接同步测试
- generate 是异步函数，其中 AsyncOpenAI 客户端需要 mock
"""
import pytest
from unittest.mock import AsyncMock, patch
from app.rag.generator import build_messages


# ============================================================
# build_messages 纯函数测试
# ============================================================

def test_build_messages_simple():
    """无历史时，只包含 system + user 两条消息"""
    messages = build_messages("参考内容", "问题", "")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "问题"


def test_build_messages_with_history():
    """有历史时，user 消息应包含对话历史"""
    history = "用户：之前的提问\n助手：之前的回答"
    messages = build_messages("参考内容", "当前问题", history)
    assert len(messages) == 2
    assert "历史对话" in messages[1]["content"]
    assert "当前问题" in messages[1]["content"]


def test_build_messages_system_prompt_contains_context():
    """system prompt 应包含核心规则和参考资料占位符"""
    messages = build_messages("测试文档内容", "问题", "")
    sys_content = messages[0]["content"]
    assert "【核心规则】" in sys_content
    assert "【参考资料】" in sys_content
    assert "测试文档内容" in sys_content


def test_build_messages_empty_context():
    """context 为空时，参考资料部分应为空"""
    messages = build_messages("", "问题", "")
    sys_content = messages[0]["content"]
    assert "【参考资料】" in sys_content
    assert sys_content.endswith("【参考资料】\n")


def test_build_messages_preserves_whitespace():
    """问题中的前后空格应保留"""
    messages = build_messages("ctx", "  问题   ", "")
    assert messages[1]["content"] == "  问题   "


# ============================================================
# generate 异步函数测试（mock OpenAI）
# ============================================================

@pytest.mark.asyncio
async def test_generate_success():
    """正常返回应得到 answer 和耗时"""
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "测试回答"

    with patch("app.rag.generator.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        from app.rag.generator import generate
        answer, cost = await generate([{"role": "user", "content": "你好"}])

    assert answer == "测试回答"
    assert cost > 0


@pytest.mark.asyncio
async def test_generate_retry_on_rate_limit():
    """限流时应自动重试，最终成功"""
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "重试后成功"

    with patch("app.rag.generator.get_client") as mock_get_client:
        mock_client = AsyncMock()
        # 前两次限流，第三次成功
        from unittest.mock import MagicMock
        from openai import RateLimitError
        mock_resp = MagicMock()
        mock_resp.request = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=mock_resp, body=None),
            RateLimitError("rate limited again", response=mock_resp, body=None),
            mock_response,
        ]
        mock_get_client.return_value = mock_client

        from app.rag.generator import generate
        answer, cost = await generate([{"role": "user", "content": "你好"}])

    assert answer == "重试后成功"
    assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_generate_all_retries_exhausted():
    """所有重试都失败时应返回友好提示"""
    with patch("app.rag.generator.get_client") as mock_get_client:
        mock_client = AsyncMock()
        from unittest.mock import MagicMock
        from openai import APIError
        mock_req = MagicMock()
        mock_client.chat.completions.create.side_effect = APIError(
            "server error", request=mock_req, body=None
        )
        mock_get_client.return_value = mock_client

        from app.rag.generator import generate
        answer, cost = await generate([{"role": "user", "content": "你好"}])

    assert "暂时不可用" in answer
    assert cost == 0.0
    assert mock_client.chat.completions.create.call_count == 3
