"""
从评测结果 JSON 生成可视化图表
用法: python experiments/generate_charts.py <results_json>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试设置中文字体
for font_name in ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "PingFang SC", "Heiti SC"]:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font_name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

CHARTS_DIR = Path(__file__).resolve().parent / "charts"


def load_results(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metrics_summary(data: dict):
    """主指标柱状图：准确召回率、模糊召回率、拒答准确率 + 平均响应时间"""
    metrics = data["metrics"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：召回率/拒答率
    categories = ["准确召回率\n(Accurate)", "模糊召回率\n(Fuzzy)", "干扰拒答率\n(Disturb Refusal)"]
    values = [
        metrics["accurate_recall"] * 100,
        metrics["fuzzy_recall"] * 100,
        metrics["refuse_accuracy"] * 100,
    ]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax1.bar(categories, values, color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("百分比 (%)", fontsize=12)
    ax1.set_title("核心指标概览", fontsize=14, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 右图：平均响应时间
    time_labels = ["检索耗时\n(Retrieve)", "LLM 耗时\n(LLM)", "总耗时\n(Total)"]
    time_values = [metrics["avg_retrieve_ms"], metrics["avg_llm_ms"], metrics["avg_total_ms"]]
    time_colors = ["#3498db", "#9b59b6", "#1abc9c"]
    bars2 = ax2.bar(time_labels, time_values, color=time_colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars2, time_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{v:.0f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylabel("耗时 (ms)", fontsize=12)
    ax2.set_title("平均响应耗时", fontsize=14, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(f"RAG 评测报告 — 总计 {metrics['accurate_count'] + metrics['fuzzy_count'] + metrics['disturb_count']} 题",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ metrics_summary.png")


def plot_recall_distribution(data: dict):
    """准确题和模糊题的召回率分布直方图"""
    accurate_recalls = [r["recall"] for r in data["results"]
                        if r["type"] == "准确" and r["recall"] is not None]
    fuzzy_recalls = [r["recall"] for r in data["results"]
                     if r["type"] == "模糊" and r["recall"] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    if accurate_recalls:
        axes[0].hist(accurate_recalls, bins=20, color="#2ecc71", alpha=0.7, edgecolor="white")
        axes[0].axvline(x=sum(accurate_recalls) / len(accurate_recalls), color="red",
                        linestyle="--", label=f"均值 {sum(accurate_recalls)/len(accurate_recalls):.2%}")
    axes[0].set_xlabel("召回率", fontsize=12)
    axes[0].set_ylabel("题数", fontsize=12)
    axes[0].set_title(f"准确题召回率分布 (n={len(accurate_recalls)})", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    if fuzzy_recalls:
        axes[1].hist(fuzzy_recalls, bins=20, color="#f39c12", alpha=0.7, edgecolor="white")
        axes[1].axvline(x=sum(fuzzy_recalls) / len(fuzzy_recalls), color="red",
                        linestyle="--", label=f"均值 {sum(fuzzy_recalls)/len(fuzzy_recalls):.2%}")
    axes[1].set_xlabel("召回率", fontsize=12)
    axes[1].set_title(f"模糊题召回率分布 (n={len(fuzzy_recalls)})", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "recall_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ recall_distribution.png")


def plot_time_distribution(data: dict):
    """响应时间分布：准确题 vs 模糊题 vs 干扰题"""
    types = {"准确": "#2ecc71", "模糊": "#f39c12", "干扰": "#e74c3c"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (metric, label) in enumerate([
        ("retrieve_cost_ms", "检索耗时 (ms)"),
        ("llm_cost_ms", "LLM 耗时 (ms)"),
        ("total_cost_ms", "总耗时 (ms)"),
    ]):
        ax = axes[idx]
        all_data = []
        labels = []
        for tname, color in types.items():
            vals = [r[metric] for r in data["results"] if r["type"] == tname]
            if vals:
                all_data.append(vals)
                labels.append(f"{tname}\n(n={len(vals)})")

        bp = ax.boxplot(all_data, patch_artist=True, widths=0.5)
        ax.set_xticklabels(labels, fontsize=9)
        for patch, color in zip(bp["boxes"], list(types.values())):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "time_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ time_distribution.png")


def plot_recall_vs_time(data: dict):
    """召回率 vs 响应时间散点图"""
    accurate = [r for r in data["results"] if r["type"] == "准确" and r["recall"] is not None]
    fuzzy = [r for r in data["results"] if r["type"] == "模糊" and r["recall"] is not None]

    fig, ax = plt.subplots(figsize=(10, 6))

    if accurate:
        ax.scatter(
            [r["total_cost_ms"] for r in accurate],
            [r["recall"] for r in accurate],
            color="#2ecc71", alpha=0.5, label=f"准确题 (n={len(accurate)})",
            edgecolors="white", linewidth=0.5,
        )
    if fuzzy:
        ax.scatter(
            [r["total_cost_ms"] for r in fuzzy],
            [r["recall"] for r in fuzzy],
            color="#f39c12", alpha=0.6, label=f"模糊题 (n={len(fuzzy)})",
            edgecolors="white", linewidth=0.5, marker="s",
        )

    ax.set_xlabel("总响应时间 (ms)", fontsize=12)
    ax.set_ylabel("召回率", fontsize=12)
    ax.set_title("召回率 vs 响应时间", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "recall_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ recall_vs_time.png")


def plot_refusal_analysis(data: dict):
    """干扰题详情：拒答正确 vs 拒答失败的耗时对比"""
    disturb = [r for r in data["results"] if r["type"] == "干扰"]

    if not disturb:
        print("  - 无干扰题数据，跳过拒答分析图")
        return

    correct = [r for r in disturb if r["is_refuse_correct"]]
    failed = [r for r in disturb if not r["is_refuse_correct"]]

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["拒答正确", "拒答失败"]
    counts = [len(correct), len(failed)]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(categories, counts, color=colors, width=0.4, edgecolor="white")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("题数", fontsize=12)
    ax.set_title(f"干扰题拒答分析 (n={len(disturb)}, "
                 f"准确率={len(correct)/len(disturb)*100:.1f}%)",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "refusal_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ refusal_analysis.png")


def main():
    if len(sys.argv) < 2:
        # 自动查找最新的评测结果文件
        results_dir = Path(__file__).resolve().parent
        json_files = sorted(results_dir.glob("eval_results_*.json"))
        if not json_files:
            print("错误：未指定评测结果 JSON 文件，且未找到 eval_results_*.json")
            sys.exit(1)
        json_path = str(json_files[-1])
        print(f"自动使用最新的评测结果: {json_path}")
    else:
        json_path = sys.argv[1]

    data = load_results(json_path)
    metrics = data["metrics"]

    total = metrics["accurate_count"] + metrics["fuzzy_count"] + metrics["disturb_count"]
    print(f"加载评测结果: {total} 题 (准确{metrics['accurate_count']} 模糊{metrics['fuzzy_count']} 干扰{metrics['disturb_count']})")
    print(f"核心指标: 准确召回{metrics['accurate_recall']:.2%} 模糊召回{metrics['fuzzy_recall']:.2%} 拒答率{metrics['refuse_accuracy']:.2%}")
    print("生成图表...")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_metrics_summary(data)
    plot_recall_distribution(data)
    plot_time_distribution(data)
    plot_recall_vs_time(data)
    plot_refusal_analysis(data)

    print(f"\n所有图表已保存至 {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
