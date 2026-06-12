"""
评测指标计算
负责 Recall@K、拒答准确率等指标
"""
from app.logger import get_logger

logger = get_logger("metrics")


def calc_recall(retrieved_sources: list[str], golden_sources: list[str]) -> float | None:
    """
    计算单条召回率
    支持完整路径包含文件名就算命中
    干扰题（golden_sources 为空）返回 None
    """
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


def is_correct_refusal(answer: str) -> bool:
    """判断是否正确拒答"""
    refuse_keywords = ["暂无相关信息", "没有相关", "无相关"]
    return any(kw in answer for kw in refuse_keywords)


def calc_metrics(results: list[dict]) -> dict:
    """计算批量评测的汇总指标"""
    accurate_recalls = []
    fuzzy_recalls = []
    disturb_count = 0
    refuse_correct = 0
    retrieve_times = []
    llm_times = []
    total_times = []

    for item in results:
        q_type = item.get("type", "准确")
        recall = item.get("recall")
        retrieve_times.append(item.get("retrieve_cost_ms", 0))
        llm_times.append(item.get("llm_cost_ms", 0))
        total_times.append(item.get("total_cost_ms", 0))

        if q_type == "干扰":
            disturb_count += 1
            if item.get("is_refuse_correct", False):
                refuse_correct += 1
        elif q_type == "模糊":
            if recall is not None:
                fuzzy_recalls.append(recall)
        else:
            if recall is not None:
                accurate_recalls.append(recall)

    avg_retrieve = sum(retrieve_times) / len(retrieve_times) if retrieve_times else 0
    avg_llm = sum(llm_times) / len(llm_times) if llm_times else 0
    avg_total = sum(total_times) / len(total_times) if total_times else 0

    return {
        "accurate_count": len(accurate_recalls),
        "accurate_recall": sum(accurate_recalls) / len(accurate_recalls) if accurate_recalls else 0,
        "fuzzy_count": len(fuzzy_recalls),
        "fuzzy_recall": sum(fuzzy_recalls) / len(fuzzy_recalls) if fuzzy_recalls else 0,
        "disturb_count": disturb_count,
        "refuse_accuracy": refuse_correct / disturb_count if disturb_count > 0 else 0,
        "avg_retrieve_ms": round(avg_retrieve, 2),
        "avg_llm_ms": round(avg_llm, 2),
        "avg_total_ms": round(avg_total, 2),
    }
