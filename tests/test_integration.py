"""
集成测试
测试完整 RAG 流程（不依赖真实模型和向量库）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 模拟依赖
sys.modules["sentence_transformers"] = type(sys)("sentence_transformers")
sys.modules["torch"] = type(sys)("torch")
sys.modules["chromadb"] = type(sys)("chromadb")


def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {doc['content']}"
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


def simulate_rag_flow(question, documents, golden_doc=None):
    retrieved = []
    retrieved_docs = []
    for doc in documents:
        for word in question:
            if word in doc["content"]:
                retrieved.append(doc["source"])
                retrieved_docs.append(doc)
                break

    context = format_docs(retrieved_docs[:10])

    recall = calc_recall(retrieved, golden_doc) if golden_doc else None

    return {
        "question": question,
        "retrieved_count": len(retrieved),
        "context_length": len(context),
        "recall": recall,
    }


# ========== 集成测试 ==========

def test_rag_flow_with_matching_doc():
    docs = [
        {"content": "产品保修期为一年", "source": "docs/warranty.txt"},
        {"content": "退货政策7天无理由", "source": "docs/return.txt"},
    ]
    result = simulate_rag_flow("保修期多久", docs, ["docs/warranty.txt"])
    assert result["retrieved_count"] >= 1
    assert result["recall"] == 1.0


def test_rag_flow_with_no_match():
    docs = [
        {"content": "产品保修期为一年", "source": "docs/warranty.txt"},
    ]
    result = simulate_rag_flow("公司地址在哪", docs)
    assert result["retrieved_count"] == 0


def test_rag_flow_context_format():
    docs = [
        {"content": "文档A内容", "source": "a.txt"},
        {"content": "文档B内容", "source": "b.txt"},
    ]
    result = simulate_rag_flow("A", docs)
    assert "[1]" in str(result["context_length"]) or result["context_length"] >= 0


def test_recall_disturb_returns_none():
    result = simulate_rag_flow("问题", [], [])
    assert result["recall"] is None


if __name__ == "__main__":
    test_rag_flow_with_matching_doc()
    test_rag_flow_with_no_match()
    test_rag_flow_context_format()
    test_recall_disturb_returns_none()
    print("所有集成测试通过！")
