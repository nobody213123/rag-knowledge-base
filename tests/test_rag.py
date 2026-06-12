"""
RAG 引擎单元测试
只测试 format_docs、calc_recall 等纯函数
不触发模块级初始化（避免加载嵌入模型）
"""
import sys

# 模拟嵌入模型和向量库，避免加载真实模型
sys.modules["sentence_transformers"] = type(sys)("sentence_transformers")
sys.modules["torch"] = type(sys)("torch")
sys.modules["chromadb"] = type(sys)("chromadb")


class FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


# 直接从源码中提取纯函数，不触发模块初始化
def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )


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


# ========== format_docs 测试 ==========

def test_format_docs_single():
    docs = [FakeDoc("这是第一个文档")]
    result = format_docs(docs)
    assert result == "[1] 这是第一个文档"


def test_format_docs_multiple():
    docs = [FakeDoc("文档A"), FakeDoc("文档B"), FakeDoc("文档C")]
    result = format_docs(docs)
    assert "[1] 文档A" in result
    assert "[2] 文档B" in result
    assert "[3] 文档C" in result
    assert "\n\n" in result


def test_format_docs_empty():
    result = format_docs([])
    assert result == ""


# ========== calc_recall 测试 ==========

def test_calc_recall_full_hit():
    retrieved = ["docs/a.txt", "docs/b.txt"]
    golden = ["a.txt", "b.txt"]
    recall = calc_recall(retrieved, golden)
    assert recall == 1.0


def test_calc_recall_partial_hit():
    retrieved = ["docs/a.txt", "docs/c.txt"]
    golden = ["a.txt", "b.txt"]
    recall = calc_recall(retrieved, golden)
    assert recall == 0.5


def test_calc_recall_no_hit():
    retrieved = ["docs/c.txt", "docs/d.txt"]
    golden = ["a.txt", "b.txt"]
    recall = calc_recall(retrieved, golden)
    assert recall == 0.0


def test_calc_recall_disturb_returns_none():
    retrieved = ["docs/a.txt"]
    golden = []
    recall = calc_recall(retrieved, golden)
    assert recall is None


def test_calc_recall_path_contains_filename():
    retrieved = ["./documents/企业知识库（数码电商产品完整版）——可直接用于RAG项目.docx"]
    golden = ["企业知识库（数码电商产品完整版）——可直接用于RAG项目.docx"]
    recall = calc_recall(retrieved, golden)
    assert recall == 1.0


if __name__ == "__main__":
    test_format_docs_single()
    test_format_docs_multiple()
    test_format_docs_empty()
    test_calc_recall_full_hit()
    test_calc_recall_partial_hit()
    test_calc_recall_no_hit()
    test_calc_recall_disturb_returns_none()
    test_calc_recall_path_contains_filename()
    print("所有测试通过！")
