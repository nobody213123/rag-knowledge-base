"""
评估指标单元测试
测试 calc_recall、is_correct_refusal 等纯函数
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 模拟依赖
sys.modules["sentence_transformers"] = type(sys)("sentence_transformers")
sys.modules["torch"] = type(sys)("torch")
sys.modules["chromadb"] = type(sys)("chromadb")


def calc_recall(retrieved_sources, golden_sources):
    hit = 0
    total = len(golden_sources)
    if total == 0:
        return None
    for gold_name in golden_sources:
        gold_name = gold_name.strip()
        for path in retrieved_sources:
            if gold_name in path:
                hit += 1
                break
    return hit / total


def is_correct_refusal(answer):
    refuse_keywords = ["暂无相关信息", "没有相关", "无相关"]
    return any(kw in answer for kw in refuse_keywords)


# ========== calc_recall 测试 ==========

def test_calc_recall_full_hit():
    assert calc_recall(["a.txt", "b.txt"], ["a.txt", "b.txt"]) == 1.0


def test_calc_recall_partial_hit():
    assert calc_recall(["a.txt", "c.txt"], ["a.txt", "b.txt"]) == 0.5


def test_calc_recall_no_hit():
    assert calc_recall(["c.txt", "d.txt"], ["a.txt", "b.txt"]) == 0.0


def test_calc_recall_disturb_returns_none():
    assert calc_recall(["a.txt"], []) is None


# ========== is_correct_refusal 测试 ==========

def test_is_correct_refusal_true():
    assert is_correct_refusal("暂无相关信息，请联系人工客服") is True


def test_is_correct_refusal_false():
    assert is_correct_refusal("产品保修期为一年") is False


def test_is_correct_refusal_partial():
    assert is_correct_refusal("没有相关文档") is True


if __name__ == "__main__":
    test_calc_recall_full_hit()
    test_calc_recall_partial_hit()
    test_calc_recall_no_hit()
    test_calc_recall_disturb_returns_none()
    test_is_correct_refusal_true()
    test_is_correct_refusal_false()
    test_is_correct_refusal_partial()
    print("所有评估指标测试通过！")
