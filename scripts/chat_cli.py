"""
CLI 交互脚本
用法: python -m scripts.chat_cli
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.pipeline import ask_with_history, clear_history


def main():
    print("正在预热向量库...")
    from app.rag.retriever import get_retriever
    get_retriever().invoke("预热")

    print("RAG 引擎已启动")
    print("输入 eval = 批量评测召回率&平均耗时&拒答率")
    print("直接输问题 = 单次问答")
    print("输入 clear = 清空对话历史")
    print("输入 quit 退出")
    print("-" * 40)

    while True:
        user_in = input("\n请输入：").strip()
        if user_in.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_in.lower() == "eval":
            from app.evaluation.runner import run_evaluation
            run_evaluation()
            continue
        if user_in.lower() == "clear":
            clear_history()
            print("对话历史已清空")
            continue
        if not user_in:
            continue

        res = ask_with_history(user_in, use_history=True)
        print(f"\n回答：{res['answer']}")
        print(f"\n对话轮数：{res['history_length']}")
        print("\n溯源信息：")
        for src in res["sources_detail"]:
            print(f"  [{src['index']}] {src['file']}")
        print(f"\n【性能】检索 {res['retrieve_cost_ms']}ms | LLM {res['llm_cost_ms']}ms | 总耗时 {res['total_cost_ms']}ms")


if __name__ == "__main__":
    main()
