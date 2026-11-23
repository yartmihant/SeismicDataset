"""Профили α слоя 175 м: прежний коэффициент, половина и четверть."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from scripts.project_paths import PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.dev_1_4_server_preparation.compare_smoke import file_evidence
from scripts.dev_1_4_server_preparation.velocity_seismograms import save_figure
from scripts.reporting import mapping, number, read_json

DATA = PROJECT_ROOT / "data/dev_1_4_server_preparation"
IMAGES = PROJECT_ROOT / "img/dev_1_4_server_preparation"
BASELINE = DATA / "homogeneous_3060_v1/padded/metadata.json"
FACTORS = (1., .5, .25)
LABELS = ("Прежний профиль · α₀", "Половина · 0,5α₀", "Четверть · 0,25α₀")
COLORS = ("#444b54", "#2466a4", "#bf5b20")


def build(output: Path, reports: Path) -> dict[str, object]:
    output.resolve().relative_to(IMAGES)
    reports.resolve().relative_to(DATA)
    if output.exists() or reports.exists():
        raise FileExistsError("Select new profile plot directories")
    baseline = read_json(BASELINE)
    alpha0 = number(mapping(baseline["run_spec"])["alpha_max_s_inv"])
    s = np.linspace(0., 175., 701)
    x = np.linspace(-175., 875., 2101)
    distance = np.maximum.reduce((-x, x-700., np.zeros_like(x)))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.6]})
    for factor, label, color in zip(FACTORS, LABELS, COLORS, strict=True):
        axes[0].plot(s, factor*alpha0*(s/175.)**2, color=color, lw=2.4,
            label=f"{label}: αmax={factor*alpha0:.2f} с⁻¹")
        axes[1].plot(x, factor*alpha0*(distance/175.)**2, color=color, lw=2.4, label=label)
    axes[0].set(xlabel="Расстояние от основной области в слой s, м", ylabel="Коэффициент α, с⁻¹",
        xlim=(0, 175), ylim=(0, alpha0*1.10), title="Поперёк слоя · α(s)=kα₀(s/175)²")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].axvspan(0., 700., color="#eceff1", zorder=0)
    for point in (0., 700.):
        axes[1].axvline(point, color="#888888", ls=":", lw=1)
    axes[1].text(350., alpha0*.8, "Основная область 700 м\nα(x,0)=0", ha="center", fontsize=12)
    axes[1].set(xlabel="Координата вдоль верхней поверхности x, м", ylabel="Коэффициент α(x,0), с⁻¹",
        xlim=(-175., 875.), ylim=(0, alpha0*1.10), xticks=[-175, 0, 175, 350, 525, 700, 875],
        title="Верх y=0: демпфируются только боковые продолжения")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Слой 175 м · меняется только сила квадратичного массового демпфирования", fontsize=14)
    fig.supxlabel("Показан коэффициент в C=αM, а не амплитуда волны. Верх остаётся свободным.\nВ нижних углах складываются боковой и нижний вклады; на верхней поверхности нижнего вклада нет.", fontsize=10)
    output.mkdir(parents=True, exist_ok=False)
    reports.mkdir(parents=True, exist_ok=False)
    figures = save_figure(fig, output, "profiles")
    record: dict[str, object] = {"status": "completed", "utc": datetime.now(timezone.utc).isoformat(),
        "padding_m": 175., "alpha_reference_s_inv": alpha0, "amplitude_factors": FACTORS,
        "profile_formula": "alpha=k*alpha0*((max(-x,x-700,0)/175)^2+(max(-y-700,0)/175)^2)",
        "peak_per_direction_s_inv": [factor*alpha0 for factor in FACTORS],
        "integral_normal_s_m_per_s": [factor*alpha0*175/3 for factor in FACTORS],
        "top_surface": "y=0 is traction-free; mass damping remains active in x<0 and x>700 extensions; alpha=0 on0<=x<=700",
        "figures": figures, "baseline_metadata": file_evidence(BASELINE), "code": file_evidence(Path(__file__)),
        "solver_run": False, "meaning": "Approved coefficient profiles, not a measured wave attenuation or reflection"}
    with (reports/"profile_plot.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=IMAGES/"profile_sweep_plan_v1")
    parser.add_argument("--report", type=Path, default=DATA/"profile_sweep_plan_v1")
    args = parser.parse_args()
    build(args.out, args.report)
    print(json.dumps({"status": "completed", "out": str(args.out)}))


if __name__ == "__main__":
    main()
