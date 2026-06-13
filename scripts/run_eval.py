"""
运行评测脚本
用法: python -m scripts.run_eval
"""
import asyncio
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.runner import run_evaluation


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 评测脚本")
    parser.add_argument(
        "--test-file",
        default="test_set_1000.json",
        help="测试集文件名（默认: test_set_1000.json）",
    )
    args = parser.parse_args()

    metrics = asyncio.run(run_evaluation(args.test_file))
    print(f"\n评测完成！准确率: {metrics['accurate_recall']:.2%}, 拒答率: {metrics['refuse_accuracy']:.2%}")


if __name__ == "__main__":
    main()
