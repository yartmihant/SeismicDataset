"""Plot saved compact coarse-order metrics without solving or loading fields."""
from __future__ import annotations

import matplotlib.pyplot as plt
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json


def main() -> None:
    report = read_json(PROJECT_ROOT / "data/dev_1_4_server_preparation/coarse_order_v1/analysis_v1/analysis.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    models = mapping(report["models"])
    orders = (3, 4, 6, 7)
    labels = ["p3 / 5 м", "p4 / 5 м", "p6 / 10 м", "p7 / 10 м"]
    for model, color in (("12404", "#1971c2"), ("65", "#d9480f")):
        data = mapping(models[model])
        pairs = mapping(data["pairs"])
        for col, metric in enumerate(("dataset70_far20/late", "dataset70/full")):
            values = [100 * number(mapping(mapping(pairs[f"p{p}/fixed"])[metric])["relative_l2"]) for p in orders]
            axes[col].plot(range(4), values, "o-", color=color, label=f"ECVB {int(model):05d}")
            for i, value in enumerate(values):
                axes[col].annotate(f"{value:.4g}", (i, value), xytext=(0, 10 if model == "65" else -17), textcoords="offset points", ha="center", fontsize=9, color=color)
        runs = mapping(data["runs"])
        times = [number(mapping(runs[f"p{p}/native"])["total_seconds"]) for p in (3, 4, 5, 6, 7)]
        axes[2].plot(range(5), times, "o-", color=color, label=f"ECVB {int(model):05d}")
        for i, value in enumerate(times):
            axes[2].annotate(f"{value:.1f}", (i, value), xytext=(0, 8), textcoords="offset points", ha="center", color=color, fontsize=9)
    for ax, title in zip(axes[:2], ("Поздняя волна: 0,5–1 с, вне 20 м", "Вся запись: 0–1 с, все 70 приёмников")):
        ax.set(xticks=range(4), xticklabels=labels, yscale="log", title=title, ylabel="Относительная L2-разность с p5 / 5 м, %")
        ax.margins(y=.25)
    axes[2].set(xticks=range(5), xticklabels=["p3/5", "p4/5", "p5/5", "p6/10", "p7/10"], title="A100: один источник, 1 с", ylabel="Время с подготовкой, с")
    axes[2].margins(y=.15)
    for ax in axes:
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("β=1/27 · разности при общем малом dt · время при native CFL=0,8\np5 — контрольное численное решение; геометрия одинакова внутри каждой новой серии")
    out = PROJECT_ROOT / "img/dev_1_4_server_preparation/coarse_order_v1/analysis_v1/precision.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
