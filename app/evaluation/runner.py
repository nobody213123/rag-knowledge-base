"""
评测运行器
负责批量跑测试集并生成报告
"""
import asyncio
import json
from pathlib import Path
from datetime import datetime
from app.config import DATA_DIR
from app.logger import get_logger
from app.rag.pipeline import ask
from app.evaluation.metrics import calc_recall, is_correct_refusal, calc_metrics

logger = get_logger("runner")


async def run_evaluation(test_file: str = "test_set_1000.json", save_results: bool = True) -> dict:
    """加载测试集，批量评测"""
    test_path = DATA_DIR / test_file
    if not test_path.exists():
        test_path = Path(test_file)

    logger.info("开始批量 RAG 评测")
    print("\n========== 开始批量 RAG 评测 ==========")

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"加载测试集: {len(test_data)} 条")

    results = []
    for idx, item in enumerate(test_data):
        q = item["question"]
        q_type = item.get("type", "准确")
        golden_doc = [item["related_doc"].strip()] if item.get("related_doc") else []

        res = await ask(q)
        recall = calc_recall(res["retrieved_sources"], golden_doc)
        is_refuse = is_correct_refusal(res["answer"])

        results.append({
            "id": item.get("id", idx),
            "question": q,
            "type": q_type,
            "golden_doc": golden_doc,
            "recall": recall,
            "is_refuse_correct": q_type == "干扰" and is_refuse,
            "retrieve_cost_ms": res["retrieve_cost_ms"],
            "llm_cost_ms": res["llm_cost_ms"],
            "total_cost_ms": res["total_cost_ms"],
        })

        print(f"\n[{idx+1}/{len(test_data)}] 【{q_type}】{q}")
        if q_type == "干扰":
            status = "拒答正确" if is_refuse else "拒答失败"
            print(f"状态：{status}")
        else:
            print(f"召回率：{recall:.2%}")
        print(f"检索：{res['retrieve_cost_ms']}ms | LLM：{res['llm_cost_ms']}ms | 总：{res['total_cost_ms']}ms")

    metrics = calc_metrics(results)
    print("\n========== 评测汇总结果 ==========")
    print(f"准确问答总数：{metrics['accurate_count']} 道 | 平均召回率：{metrics['accurate_recall']:.2%}")
    print(f"模糊问答总数：{metrics['fuzzy_count']} 道 | 平均召回率：{metrics['fuzzy_recall']:.2%}")
    print(f"干扰问答总数：{metrics['disturb_count']} 道 | 拒答准确率：{metrics['refuse_accuracy']:.2%}")
    print(f"全局平均检索耗时：{metrics['avg_retrieve_ms']}ms")
    print(f"全局平均LLM耗时：{metrics['avg_llm_ms']}ms")
    print(f"全局平均总响应耗时：{metrics['avg_total_ms']}ms")
    print("==================================\n")

    # 保存每条结果到 JSON 文件，供图表生成使用
    if save_results:
        experiments_dir = Path(__file__).resolve().parent.parent.parent / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = experiments_dir / f"eval_results_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": {
                    "accurate_recall": metrics["accurate_recall"],
                    "fuzzy_recall": metrics["fuzzy_recall"],
                    "refuse_accuracy": metrics["refuse_accuracy"],
                    "avg_retrieve_ms": metrics["avg_retrieve_ms"],
                    "avg_llm_ms": metrics["avg_llm_ms"],
                    "avg_total_ms": metrics["avg_total_ms"],
                    "accurate_count": metrics["accurate_count"],
                    "fuzzy_count": metrics["fuzzy_count"],
                    "disturb_count": metrics["disturb_count"],
                },
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"评测结果已保存至: {result_file}")
        print(f"\n评测结果已保存至: {result_file}")

    logger.info(f"评测完成: {metrics}")
    return metrics
