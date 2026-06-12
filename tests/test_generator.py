"""
生成器单元测试
测试实际 app.rag.generator 中的 build_messages
"""
from app.rag.generator import build_messages


# ========== build_messages 测试 ==========

def test_build_messages_simple():
    messages = build_messages("参考内容", "问题", "")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "问题"


def test_build_messages_with_history():
    history = "用户：之前的提问\n助手：之前的回答"
    messages = build_messages("参考内容", "当前问题", history)
    assert len(messages) == 2
    assert "历史对话" in messages[1]["content"]
    assert "当前问题" in messages[1]["content"]
    assert "当前问题" in messages[1]["content"]


def test_build_messages_system_prompt_contains_context():
    messages = build_messages("测试文档内容", "问题", "")
    sys_content = messages[0]["content"]
    assert "【核心规则】" in sys_content
    assert "【参考资料】" in sys_content
    assert "测试文档内容" in sys_content


def test_build_messages_empty_context():
    messages = build_messages("", "问题", "")
    sys_content = messages[0]["content"]
    assert "【参考资料】" in sys_content
    assert sys_content.endswith("【参考资料】\n")


def test_build_messages_question_trimming():
    messages = build_messages("ctx", "  问题   ", "")
    assert messages[1]["content"] == "  问题   "
