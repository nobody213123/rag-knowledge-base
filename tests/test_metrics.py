"""
评估指标单元测试
测试实际 app.evaluation.metrics 模块中的函数
"""
from app.evaluation.metrics import calc_recall, is_correct_refusal


# ========== calc_recall 测试 ==========

def test_calc_recall_full_hit():
    assert calc_recall(["a.txt", "b.txt"], ["a.txt", "b.txt"]) == 1.0


def test_calc_recall_partial_hit():
    assert calc_recall(["a.txt", "c.txt"], ["a.txt", "b.txt"]) == 0.5


def test_calc_recall_no_hit():
    assert calc_recall(["c.txt", "d.txt"], ["a.txt", "b.txt"]) == 0.0


def test_calc_recall_disturb_returns_none():
    assert calc_recall(["a.txt"], []) is None


def test_calc_recall_path_contains_filename():
    result = calc_recall(
        ["./documents/产品手册.pdf"],
        ["产品手册.pdf"],
    )
    assert result == 1.0


def test_calc_recall_duplicate_golden():
    result = calc_recall(["a.txt"], ["a.txt", "a.txt"])
    assert result == 1.0


# ========== is_correct_refusal 测试 ==========

def test_is_correct_refusal_true():
    assert is_correct_refusal("暂无相关信息，请联系人工客服") is True


def test_is_correct_refusal_false():
    assert is_correct_refusal("产品保修期为一年") is False


def test_is_correct_refusal_partial():
    assert is_correct_refusal("没有相关文档能回答这个问题") is True


def test_is_correct_refusal_empty():
    assert is_correct_refusal("") is False


def test_is_correct_refusal_keyword_at_start():
    assert is_correct_refusal("无相关资料") is True
