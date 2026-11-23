"""Add p5/10m to the saved p6/p7 convergence series; never run the solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from scripts.dev_1_4_server_preparation.kim_convergence_analysis import load, metrics
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    images = PROJECT_ROOT / "img/dev_1_4_server_preparation/coarse_order_v1" / out.name
    images.mkdir(parents=True, exist_ok=False)
    models: dict[str, object] = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for row, (model, k) in enumerate(((12404, 62), (65, 140))):
        base = args.root / f"{model:05d}"
        fixed = {p: load(base / "coarse" / f"p{p}_beta{1/27:.8f}_fixedk{k}") for p in (5, 6, 7)}
        native = {p: load(base / "coarse" / f"p{p}_beta{1/27:.8f}_cfl0.8") for p in (5, 6, 7)}
        reference = load(base / "refined" / f"p5_beta{1/27:.8f}_fixedk{k}")
        assert len({v.meta["dt_s"] for v in [*fixed.values(), reference]}) == 1
        assert len({mapping(v.meta["mesh"])["sha256"] for v in [*fixed.values(), *native.values()]}) == 1
        assert len({mapping(v.meta["native"])["sha256"] for v in [*fixed.values(), *native.values(), reference]}) == 1
        pairs = {f"p{p}/reference": metrics(fixed[p], reference) for p in (5, 6, 7)}
        pairs.update({f"p{p}/dt": metrics(native[p], fixed[p]) for p in (5, 6, 7)})
        pairs.update({"p5-p6": metrics(fixed[5], fixed[6]), "p6-p7": metrics(fixed[6], fixed[7])})
        adjacent: dict[str, object] = {}
        for key, ref_value in pairs["p5/reference"].items():
            norm = number(mapping(ref_value)["reference_l2"])
            first = mapping(pairs["p5-p6"][key])
            second = mapping(pairs["p6-p7"][key])
            d56 = number(first["relative_l2"]) * number(first["reference_l2"]) / norm
            d67 = number(second["relative_l2"]) * number(second["reference_l2"]) / norm
            adjacent[key] = {"p5_p6_over_ref": d56, "p6_p7_over_ref": d67, "contraction_ratio": d67 / d56}
        models[str(model)] = {
            "common_k": k, "pairs": pairs, "adjacent_common_normalization": adjacent,
            "runs": {f"p{p}/{label}": v.meta for label, runs in (("native", native), ("fixed", fixed)) for p, v in runs.items()},
            "reference": reference.meta,
        }
        for col, window in enumerate(("full", "late")):
            ax = axes[row, col]
            for mask, title, style in (("dataset70_far20", "66 приёмников вне 20 м", "o-"), ("dataset70", "все 70 приёмников", "s--")):
                values = [100 * number(mapping(pairs[f"p{p}/reference"][f"{mask}/{window}"])["relative_l2"]) for p in (5, 6, 7)]
                ax.plot([5, 6, 7], values, style, label=title)
                if mask == "dataset70_far20":
                    for p, value in zip((5, 6, 7), values):
                        ax.annotate(f"{value:.4g}%", (p, value), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9)
            ax.set(xticks=[5, 6, 7], xlabel="Порядок p при h≈10 м", yscale="log", ylabel="L2-разность с p5 / 5 м, %", title=f"ECVB {model:05d}: " + ("0–1 с" if window == "full" else "0,5–1 с"))
            ax.margins(x=.15, y=.25)
            ax.grid(alpha=.25, which="both")
            ax.legend(fontsize=8)
        ax = axes[row, 2]
        for label, records in (("native CFL=0,8", native), (f"общий малый dt, K={k}", fixed)):
            times = [number(records[p].meta["total_seconds"]) for p in (5, 6, 7)]
            ax.plot([5, 6, 7], times, "o-", label=label)
            for p, value in zip((5, 6, 7), times):
                ax.annotate(f"{value:.1f}", (p, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
        ax.set(xticks=[5, 6, 7], xlabel="Порядок p при h≈10 м", ylabel="Время с подготовкой, с", title=f"ECVB {model:05d}: A100, один источник")
        ax.margins(y=.2)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("p5 → p6 → p7 на одной сетке ≈10 м, β=1/27, длительность 1 с\nРазности при одинаковом малом dt; контроль p5 на точном разбиении ≈5 м не является точным решением")
    fig.savefig(images / "convergence.png", dpi=160)
    plt.close(fig)
    report = {"models": models, "scope": "Four added p5/coarse runs; existing p6/p7 and p5/refined reused with SHA verification; no solver run", "adjacent_note": "Both adjacent differences normalized to the same refined p5 norm; contraction is descriptive, not proof of convergence"}
    (out / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
