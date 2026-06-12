"""
RAG 参数对比实验 — 仿真探索脚本

⚠️ 重要声明
本脚本使用 simulate_experiment() 生成**模拟数据**，而非真实 RAG 评测结果。
目的是在缺乏真实评测流水线的情况下，快速探索参数变化对系统指标的
影响趋势（如 chunk_size 增大则召回率下降、检索耗时增加等定量关系）。

真实评测结果请参考 README.md 中的 83.67% 召回率数据。
"""
import csv
from pathlib import Path

RESULTS_DIR = Path("./results")
CHARTS_DIR = Path("./charts")
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)


def simulate_experiment(param_name, param_values):
    """
    模拟参数对比实验数据（非真实 RAG 评测结果）

    基于以下先验知识生成"合理"的趋势数据：
    1. chunk_size 过大 → 语义被稀释 → 召回率下降
    2. overlap 过小 → 跨块句断裂 → 召回率下降
    3. lambda_mult 偏离 0.7 过多 → 相似度/多样性失衡
    4. k 太大 → 引入噪声 → 精度下降，耗时增加

    真实场景中应替换为 run_eval.py 的实际评测流水线。
    """
    results = []

    for value in param_values:
        if param_name == "chunk_size":
            if value == 100:
                acc, fuzzy, refuse = 82.0, 75.0, 78.0
                retrieve, total = 25.0, 3200.0
            elif value == 200:
                acc, fuzzy, refuse = 83.0, 77.0, 80.0
                retrieve, total = 35.0, 3500.0
            elif value == 500:
                acc, fuzzy, refuse = 78.0, 70.0, 76.0
                retrieve, total = 50.0, 3800.0
            elif value == 1000:
                acc, fuzzy, refuse = 72.0, 65.0, 72.0
                retrieve, total = 80.0, 4200.0
            else:
                acc, fuzzy, refuse = 65.0, 58.0, 68.0
                retrieve, total = 120.0, 4800.0

        elif param_name == "overlap":
            if value == 0:
                acc, fuzzy, refuse = 76.0, 68.0, 78.0
                retrieve, total = 30.0, 3300.0
            elif value == 30:
                acc, fuzzy, refuse = 81.0, 74.0, 79.0
                retrieve, total = 32.0, 3400.0
            elif value == 50:
                acc, fuzzy, refuse = 83.0, 77.0, 80.0
                retrieve, total = 35.0, 3500.0
            elif value == 100:
                acc, fuzzy, refuse = 83.0, 76.0, 80.0
                retrieve, total = 40.0, 3600.0
            else:
                acc, fuzzy, refuse = 82.0, 74.0, 79.0
                retrieve, total = 45.0, 3700.0

        elif param_name == "lambda_mult":
            if value == 0.5:
                acc, fuzzy, refuse = 76.0, 68.0, 78.0
                retrieve, total = 38.0, 3550.0
            elif value == 0.6:
                acc, fuzzy, refuse = 80.0, 74.0, 79.0
                retrieve, total = 36.0, 3520.0
            elif value == 0.7:
                acc, fuzzy, refuse = 83.0, 77.0, 80.0
                retrieve, total = 35.0, 3500.0
            elif value == 0.8:
                acc, fuzzy, refuse = 82.0, 75.0, 79.5
                retrieve, total = 34.0, 3480.0
            else:
                acc, fuzzy, refuse = 78.0, 70.0, 78.0
                retrieve, total = 33.0, 3450.0

        elif param_name == "k":
            if value == 5:
                acc, fuzzy, refuse = 74.0, 66.0, 76.0
                retrieve, total = 20.0, 3100.0
            elif value == 10:
                acc, fuzzy, refuse = 83.0, 77.0, 80.0
                retrieve, total = 35.0, 3500.0
            elif value == 15:
                acc, fuzzy, refuse = 83.0, 76.0, 79.0
                retrieve, total = 50.0, 3800.0
            elif value == 20:
                acc, fuzzy, refuse = 81.0, 73.0, 78.0
                retrieve, total = 65.0, 4100.0
            else:
                acc, fuzzy, refuse = 78.0, 68.0, 76.0
                retrieve, total = 90.0, 4500.0

        results.append({
            "param": param_name,
            "value": value,
            "accuracy": acc,
            "fuzzy_rate": fuzzy,
            "refuse_rate": refuse,
            "retrieve_ms": retrieve,
            "total_ms": total,
        })

    return results


def save_csv(results, filename):
    """保存结果为 CSV 文件"""
    filepath = RESULTS_DIR / filename
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"已保存: {filepath}")


def generate_charts():
    """从 CSV 生成可视化图表（需要 matplotlib）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv
    except ImportError:
        print("matplotlib 未安装，跳过图表生成")
        print("运行: pip install matplotlib")
        return

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    experiments = [
        ("chunk_size", "chunk_size.csv", "分块大小对系统效果的影响"),
        ("overlap", "overlap.csv", "重叠大小对系统效果的影响"),
        ("lambda_mult", "lambda_mult.csv", "MMR lambda 对系统效果的影响"),
        ("k", "k.csv", "检索数量 k 对系统效果的影响"),
    ]

    for param, csv_file, title in experiments:
        data = []
        with open(RESULTS_DIR / csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        values = [d["value"] for d in data]
        accuracy = [float(d["accuracy"]) for d in data]
        fuzzy = [float(d["fuzzy_rate"]) for d in data]
        refuse = [float(d["refuse_rate"]) for d in data]
        retrieve = [float(d["retrieve_ms"]) for d in data]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.plot(values, accuracy, "o-", label="准确题召回率", linewidth=2)
        ax1.plot(values, fuzzy, "s-", label="模糊题召回率", linewidth=2)
        ax1.plot(values, refuse, "^-", label="干扰题拒答率", linewidth=2)
        ax1.set_xlabel(param)
        ax1.set_ylabel("百分比 (%)")
        ax1.set_title(f"{title} - 准确率")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(50, 90)

        ax2 = axes[1]
        ax2.plot(values, retrieve, "D-", color="red", linewidth=2)
        ax2.set_xlabel(param)
        ax2.set_ylabel("检索耗时 (ms)")
        ax2.set_title(f"{title} - 检索耗时")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = CHARTS_DIR / f"{param}_impact.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"已生成图表: {chart_path}")


def main():
    """运行四组参数对比实验（仿真数据）"""
    print("=" * 60)
    print("RAG 参数对比实验（仿真数据）")
    print("=" * 60)
    print("注意：本实验使用模拟数据，仅用于探索参数影响趋势")
    print("真实评测结果请参考 README.md\n")

    experiments = {
        "chunk_size": [100, 200, 500, 1000, 2000],
        "overlap": [0, 30, 50, 100, 150],
        "lambda_mult": [0.5, 0.6, 0.7, 0.8, 1.0],
        "k": [5, 10, 15, 20, 30],
    }

    for param, values in experiments.items():
        print(f"\n[实验] 测试 {param}: {values}")
        results = simulate_experiment(param, values)
        csv_file = f"{param}.csv"
        save_csv(results, csv_file)

    print("\n[图表] 生成可视化图表...")
    generate_charts()

    print("\n" + "=" * 60)
    print("实验完成！")
    print(f"数据目录: {RESULTS_DIR}")
    print(f"图表目录: {CHARTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
