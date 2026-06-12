"""
评测运行器
负责批量跑测试集并生成报告
"""
import json
from pathlib import Path
from app.config import DATA_DIR
from app.logger import get_logger
from app.rag.pipeline import ask
from app.evaluation.metrics import calc_recall, is_correct_refusal, calc_metrics

logger = get_logger("runner")


def run_evaluation(test_file: str = "test_set_400.json") -> dict:
    """加载测试集，批量评测"""
    test_path = DATA_DIR / test_file
    if not test_path.exists():
        # 兼容旧路径
        test_path = Path(test_file)

    logger.info("开始批量 RAG 评测")
    print("\n========== 开始批量 RAG 评测 ==========")

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"加载测试集: {len(test_data)} 条")

    results = []
    for item in test_data:
        q = item["question"]
        q_type = item.get("type", "准确")
        golden_doc = [item["related_doc"].strip()] if item.get("related_doc") else []

        res = ask(q)
        recall = calc_recall(res["retrieved_sources"], golden_doc)
        is_refuse = is_correct_refusal(res["answer"])

        results.append({
            "question": q,
            "type": q_type,
            "recall": recall,
            "is_refuse_correct": q_type == "干扰" and is_refuse,
            "retrieve_cost_ms": res["retrieve_cost_ms"],
            "llm_cost_ms": res["llm_cost_ms"],
            "total_cost_ms": res["total_cost_ms"],
        })

        # 打印单条结果
        print(f"\n【问题】{q}")
        if q_type == "干扰":
            status = "拒答正确" if is_refuse else "拒答失败"
            print(f"本条类型：干扰题 | {status}")
        elif q_type == "模糊":
            print(f"本条类型：模糊题 | 召回率：{recall:.2%}")
        else:
            print(f"本条类型：准确题 | 召回率：{recall:.2%}")
        print(f"检索耗时：{res['retrieve_cost_ms']}ms | LLM耗时：{res['llm_cost_ms']}ms | 总耗时：{res['total_cost_ms']}ms")

    # 汇总
    metrics = calc_metrics(results)
    print("\n========== 评测汇总结果 ==========")
    print(f"准确问答总数：{metrics['accurate_count']} 道 | 平均召回率：{metrics['accurate_recall']:.2%}")
    print(f"模糊问答总数：{metrics['fuzzy_count']} 道 | 平均召回率：{metrics['fuzzy_recall']:.2%}")
    print(f"干扰问答总数：{metrics['disturb_count']} 道 | 拒答准确率：{metrics['refuse_accuracy']:.2%}")
    print(f"全局平均检索耗时：{metrics['avg_retrieve_ms']}ms")
    print(f"全局平均LLM耗时：{metrics['avg_llm_ms']}ms")
    print(f"全局平均总响应耗时：{metrics['avg_total_ms']}ms")
    print("==================================\n")

    logger.info(f"评测完成: {metrics}")
    return metrics
