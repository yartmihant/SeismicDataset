"""Причинный анализ однородных опытов 0/175/350 м и сохранённого профиля175."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from scripts.project_paths import PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from scripts.dev_1_4_server_preparation.compare_smoke import Surface, file_evidence, metrics
from scripts.dev_1_4_server_preparation.homogeneous_visualize import MATERIAL, load_surface, validate_pair
from scripts.dev_1_4_server_preparation.velocity_seismograms import save_figure
from scripts.reporting import mapping, number, read_json

FloatArray = NDArray[np.float64]
DATA = PROJECT_ROOT / "data/dev_1_4_server_preparation"
IMAGES = PROJECT_ROOT / "img/dev_1_4_server_preparation"
VP = 4087.
VS = 2431.62036
CENTER = .05
NAMES = ("zero", "single", "double", "held")
LABELS = ("0 м", "175 м", "350 м, плавный", "350 м, профиль175 + плато")
COLORS = ("#606771", "#336ba9", "#d56720", "#6d46a4")
PAIRS = (("single_minus_zero", 1, 0, "175 − 0"),
         ("double_minus_single", 2, 1, "350 − 175"),
         ("held_minus_single", 3, 1, "Плато − 175"),
         ("double_minus_held", 2, 3, "350 − плато"))


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    metadata: dict[str, object]
    surface: Surface


def rayleigh_speed(vp: float = VP, vs: float = VS) -> float:
    """Ненулевой субсдвиговый корень детерминанта свободной поверхности."""
    beta = (vs / vp) ** 2
    root = float(brentq(lambda q: (2 - q*q)**2 - 4*np.sqrt(1-q*q)*np.sqrt(1-beta*q*q), .1, .999999))
    return root * vs


def load_run(name: str, directory: Path) -> Run:
    record = read_json(directory / "metadata.json")
    if record.get("status") != "completed" or (directory / "metadata.failed.json").exists():
        raise ValueError(f"Need completed run: {directory}")
    if record.get("physical_model") != MATERIAL or record.get("is_original_dataset_model") is not False:
        raise ValueError("Not the requested homogeneous material")
    spec = mapping(record["run_spec"])
    for key, value in {"t_end_s": 2., "substeps": 96, "order": 4, "source_x_m": 350.}.items():
        if spec.get(key) != value:
            raise ValueError(f"Wrong run parameter {key}")
    if record.get("steps_count") != 192000:
        raise ValueError("Wrong step count")
    return Run(name, directory, record, load_surface(directory, record))


def first_sustained(times: FloatArray, amplitude: FloatArray, level: float, count: int = 3) -> float | None:
    """Первое превышение порога count последовательных отсчётов; не фронт носителя."""
    above = amplitude > level
    indices = np.flatnonzero(np.convolve(above.astype(np.int64), np.ones(count, dtype=np.int64), "valid") == count)
    return float(times[indices[0]]) if len(indices) else None


def path_length(x: FloatArray, padding: float) -> FloatArray:
    """Кратчайший отражённый путь от источника до боковой/нижней внешней грани."""
    return np.minimum.reduce((350 + x + 2*padding, 1050 - x + 2*padding,
                              np.sqrt((x-350)**2 + (2*(700+padding))**2)))


def beam(times: FloatArray, xs: FloatArray, velocity: FloatArray, speed: float,
         max_x_m: float = 315.) -> tuple[FloatArray, FloatArray]:
    """Среднее по левым приёмникам после t=τ+x/cR; не модовое разложение."""
    tau = np.linspace(.0, .85, 851)
    selected = np.flatnonzero((xs >= 65) & (xs <= max_x_m))
    if len(selected) == 0:
        raise ValueError("Empty beam aperture")
    values = np.zeros((len(tau), 2), dtype=np.float64)
    for index in selected:
        for component in range(2):
            values[:, component] += np.interp(tau + xs[index]/speed, times, velocity[:, index, component])
    return tau, values / len(selected)


def diagnostics(runs: list[Run]) -> tuple[dict[str, object], list[FloatArray], list[FloatArray]]:
    times = runs[0].surface.times
    xs = runs[0].surface.coordinates[:, 0]
    speed = rayleigh_speed()
    differences = [runs[a].surface.velocity-runs[b].surface.velocity for _, a, b, _ in PAIRS]
    beams: list[FloatArray] = []
    report: dict[str, object] = {}
    direct_window_end = CENTER+np.abs(xs-350)/speed+.06
    direct_window_mask = times[:, None] <= direct_window_end[None, :]
    reference_amplitude = np.linalg.norm(runs[1].surface.velocity, axis=2)
    direct_peak = np.max(np.where(direct_window_mask, reference_amplitude, 0.), axis=0)
    for (name, a, b, label), difference in zip(PAIRS, differences, strict=True):
        tau, stacked = beam(times, xs, difference, speed)
        beams.append(stacked)
        amplitude = np.linalg.norm(difference, axis=2)
        rms = np.sqrt(np.mean(difference**2, axis=(1, 2)))
        power = np.sum(difference**2, axis=(1, 2))
        cumulative = np.cumsum(power) / np.sum(power)
        onset: list[dict[str, object]] = []
        for x in (5., 175., 345.):
            i = int(np.flatnonzero(xs == x)[0])
            trace_peak = float(np.max(amplitude[:, i]))
            p_limit = float(path_length(np.array([x]), 175.)[0])/VP
            before_old_boundary = times < p_limit-.003
            onset.append({"receiver_x_m": x, "peak_vector_difference_m_s": trace_peak,
                "peak_time_s": float(times[np.argmax(amplitude[:, i])]),
                "relative_to_own_peak": {str(f): first_sustained(times, amplitude[:, i], f*trace_peak) for f in (.001, .01, .05)},
                "reference_peak_window_s": [0., float(direct_window_end[i])],
                "reference_window_peak_m_s": float(direct_peak[i]),
                "relative_to_reference_direct_window_peak": {str(f): first_sustained(times, amplitude[:, i], f*float(direct_peak[i])) for f in (1e-6, 1e-4, 1e-3, 1e-2)},
                "rayleigh_return_peak_estimates_s": {str(p): CENTER + float(path_length(np.array([x]), p)[0])/speed for p in (0., 175., 350.)},
                "earliest_p_path_from_175_boundary_s": p_limit,
                "pre_old175_boundary_p_margin_s": .003,
                "pre_old175_boundary_p_max_vector_m_s": float(np.max(amplitude[before_old_boundary, i])),
                "pre_old175_boundary_p_max_over_trace_peak": float(np.max(amplitude[before_old_boundary, i]))/trace_peak})
        peak_windows: dict[str, object] = {}
        stacked_amplitude = np.linalg.norm(stacked, axis=1)
        for padding in (0., 175., 350.):
            center = CENTER + (350+2*padding)/speed
            window = np.abs(tau-center) <= .06
            idx = np.flatnonzero(window)[int(np.argmax(stacked_amplitude[window]))]
            peak_windows[str(padding)] = {"tau_prediction_s": center, "tau_interval_s": [center-.06, center+.06],
                "tau_peak_s": float(tau[idx]), "peak_vector_m_s": float(stacked_amplitude[idx])}
        sensitivity: list[dict[str, object]] = []
        for xmax in (175., 225., 275., 315.):
            selected_tau, selected_beam = beam(times, xs, difference, speed, xmax)
            selected_amplitude = np.linalg.norm(selected_beam, axis=1)
            early_window = np.abs(selected_tau-(CENTER+350/speed)) <= .06
            early_ids = np.flatnonzero(early_window)
            early_index = int(early_ids[np.argmax(selected_amplitude[early_window])])
            all_window = (selected_tau >= .08) & (selected_tau <= .7)
            all_ids = np.flatnonzero(all_window)
            all_index = int(all_ids[np.argmax(selected_amplitude[all_window])])
            sensitivity.append({"receiver_interval_m": [65., xmax],
                "receiver_count": int(np.count_nonzero((xs >= 65) & (xs <= xmax))),
                "early_tau_peak_s": float(selected_tau[early_index]),
                "early_peak_vector_m_s": float(selected_amplitude[early_index]),
                "early_peak_at_window_edge": bool(early_index in (early_ids[0], early_ids[-1])),
                "full_tau_interval_s": [.08, .7], "full_tau_peak_s": float(selected_tau[all_index]),
                "full_peak_vector_m_s": float(selected_amplitude[all_index])})
        report[name] = {"label": label, "metrics": metrics(runs[a].surface.velocity, runs[b].surface.velocity),
            "rms_peak_time_s": float(times[np.argmax(rms)]), "rms_peak_m_s": float(np.max(rms)),
            "difference_squared_sum_quantiles_s": {str(f): float(times[np.searchsorted(cumulative, f)]) for f in (.1, .5, .9)},
            "onsets": onset, "beam_peaks_in_predicted_windows": peak_windows,
            "beam_aperture_sensitivity": sensitivity}
    residuals: dict[str, object] = {}
    for start, end in ((.3, .8), (.8, 2.), (.3, 2.), (1., 2.)):
        mask = (times >= start) & (times <= end)
        records: dict[str, object] = {}
        norms = [float(np.linalg.norm(run.surface.velocity[mask])) for run in runs]
        for run, norm in zip(runs, norms, strict=True):
            records[run.name] = {"l2_velocity_samples_m_s": norm, "rms_velocity_m_s": float(np.sqrt(np.mean(run.surface.velocity[mask]**2))),
                "norm_relative_to_single": norm/norms[1], "squared_sum_relative_to_single": (norm/norms[1])**2}
        residuals[f"{start:g}_{end:g}"] = records
    receiver_returns: list[dict[str, object]] = []
    for x in (5., 175., 345.):
        i = int(np.flatnonzero(xs == x)[0])
        direct_r = CENTER+abs(x-350)/speed
        intervals: dict[str, tuple[float, float]] = {"after_direct_plus80ms": (direct_r+.08, 2.)}
        for padding in (0., 175., 350.):
            peak = CENTER+float(path_length(np.array([x]), padding)[0])/speed
            intervals[f"rayleigh_{padding:g}_plusminus60ms"] = (peak-.06, peak+.06)
        windows: dict[str, object] = {}
        for name, (start, end) in intervals.items():
            mask = (times >= start) & (times <= end)
            indices = np.flatnonzero(mask)
            records = {}
            for run in runs:
                values = run.surface.velocity[mask, i, :]
                magnitude = np.linalg.norm(values, axis=1)
                records[run.name] = {"max_vector_m_s": float(np.max(magnitude)),
                    "peak_time_s": float(times[indices[np.argmax(magnitude)]]),
                    "rms_vector_m_s": float(np.sqrt(np.mean(np.sum(values**2, axis=1)))),
                    "l2_vector_samples_m_s": float(np.linalg.norm(values))}
            windows[name] = {"interval_s": [start, end], "runs": records}
        receiver_returns.append({"receiver_x_m": x, "direct_rayleigh_peak_estimate_s": direct_r, "windows": windows})
    profile_part, boundary_part, total_part = differences[3], differences[2], differences[1]
    decomposition: dict[str, object] = {
        "definition": "A = stretched350 - held175plateau (profile); B = held175plateau - single175 (old outer ABC replaced by added region); total=A+B",
        "identity_max_abs_m_s": float(np.max(np.abs(profile_part+boundary_part-total_part))),
        "identity_relative_l2": float(np.linalg.norm(profile_part+boundary_part-total_part)/np.linalg.norm(total_part))}
    decomp_windows: dict[str, object] = {}
    for start, end in ((0., 2.), (0., .3), (.3, .8), (.8, 2.)):
        mask = (times >= start) & (times <= end)
        na, nb, nt = (float(np.linalg.norm(item[mask])) for item in (profile_part,boundary_part,total_part))
        inner = float(np.sum(profile_part[mask]*boundary_part[mask]))
        decomp_windows[f"{start:g}_{end:g}"] = {"profile_l2_m_s": na, "boundary_l2_m_s": nb, "total_l2_m_s": nt,
            "profile_over_boundary_l2": na/nb, "profile_max_abs_m_s": float(np.max(np.abs(profile_part[mask]))),
            "boundary_max_abs_m_s": float(np.max(np.abs(boundary_part[mask]))),
            "normalized_inner_product": inner/(na*nb), "quadratic_identity_residual": (na*na+nb*nb+2*inner-nt*nt)/(nt*nt)}
    decomposition["windows"] = decomp_windows
    held_difference = differences[2]
    p_times = path_length(xs, 175.) / VP
    pre_mask = times[:, None] < p_times[None, :] - .003
    post_mask = ~pre_mask
    pre_values = held_difference[pre_mask]
    after_values = held_difference[post_mask]
    causal = {"boundary": "old ABC at x=-175,875 and y=-875", "margin_before_p_path_s": .003,
        "pre_p_path_max_abs_m_s": float(np.max(np.abs(pre_values))),
        "post_p_path_max_abs_m_s": float(np.max(np.abs(after_values))),
        "pre_over_post_max_abs": float(np.max(np.abs(pre_values))/np.max(np.abs(after_values))),
        "pre_over_post_l2": float(np.linalg.norm(pre_values)/np.linalg.norm(after_values)),
        "meaning": "Matched old profile and mesh imply outer-boundary/added-region causality; finite SEM stencil has tiny numerical precursors, so this is a magnitude check, not exact compact-support equality"}
    early_changed_profile: dict[str, object] = {}
    for key, difference in (("double_minus_single", differences[1]), ("double_minus_held", differences[3])):
        early = difference[pre_mask]
        early_changed_profile[key] = {"pre_old175_boundary_p_max_abs_m_s": float(np.max(np.abs(early))),
            "full_max_abs_m_s": float(np.max(np.abs(difference))),
            "pre_old175_boundary_p_squared_sum_fraction": float(np.sum(early**2)/np.sum(difference**2)),
            "margin_before_p_path_s": .003,
            "meaning": "Magnitude before any continuum P-wave return from the old175 outer boundary; non-negligible values cannot be explained solely by moving that boundary"}
    pre_boundary_parts: dict[str, object] = {}
    for (key, _, _, _), difference in zip(PAIRS, differences, strict=True):
        early_norm = float(np.linalg.norm(difference[pre_mask]))
        full_norm = float(np.linalg.norm(difference))
        pre_boundary_parts[key] = {"pre_max_abs_m_s": float(np.max(np.abs(difference[pre_mask]))),
            "pre_l2_m_s": early_norm, "full_l2_m_s": full_norm,
            "pre_l2_relative_full": early_norm/full_norm,
            "pre_squared_sum_fraction": (early_norm/full_norm)**2}
    pre_profile_norm = float(np.linalg.norm(profile_part[pre_mask]))
    pre_boundary_norm = float(np.linalg.norm(boundary_part[pre_mask]))
    return {"pairs": report, "surface_tail_amplitude_proxies": residuals, "actual_receiver_return_windows": receiver_returns,
        "profile_boundary_decomposition": decomposition, "held_profile_causality": causal,
        "changed_profile_before_old_boundary_p_path": early_changed_profile,
        "pre_old_boundary_comparison": {"parts": pre_boundary_parts,
            "pre_boundary_B_over_pre_profile_A_l2": pre_boundary_norm/pre_profile_norm,
            "margin_before_p_path_s": .003,
            "mask": "time < min(left/right/bottom reflected path from source via old175 ABC to receiver)/Vp - 0.003"},
        "rayleigh_speed_m_s": speed, "rayleigh_outer_shift_175_to350_s": 350/speed,
        "p_outer_shift_175_to350_s": 350/VP}, differences, beams


def figures(runs: list[Run], differences: list[FloatArray], beams: list[FloatArray], directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    times = runs[0].surface.times
    xs = runs[0].surface.coordinates[:, 0]
    speed = rayleigh_speed()
    maximum = max(float(np.max(np.abs(d))) for d in differences)*1e9
    norm = SymLogNorm(linthresh=.001, vmin=-maximum, vmax=maximum, base=10)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True, layout="constrained")
    mesh = None
    for c in range(2):
        for j, ((_, _, _, label), difference) in enumerate(zip(PAIRS, differences, strict=True)):
            ax = axes[c, j]
            mesh = ax.pcolormesh(xs, times, difference[:, :, c]*1e9, shading="nearest", cmap="RdBu_r", norm=norm, rasterized=True)
            for pad, color, style in ((0., "#111111", "-"), (175., "#008444", "--"), (350., "#ad0077", ":")):
                ax.plot(xs, CENTER+(350+xs+2*pad)/speed, color=color, ls=style, lw=1.15, label=f"R от границы L={pad:g}")
                ax.plot(xs, CENTER+(1050-xs+2*pad)/speed, color=color, ls=style, lw=1.15)
            ax.set(xlim=(0, 700), ylim=(.8, 0), title=f"{'Vx' if c==0 else 'Vy'} · {label}")
            if j == 0:
                ax.set_ylabel("Время, с")
            if c == 1:
                ax.set_xlabel("x, м")
    axes[0, 0].legend(loc="upper center", fontsize=8, framealpha=.92)
    if mesh is None:
        raise AssertionError("No image")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=.8, label="Δv, нм/с · общая SymLog-шкала; линейно до 0,001")
    fig.suptitle("Однородная среда · разности четырёх постановок\nЛинии — кинематические оценки пиков возврата Релея; не выделенные моды", fontsize=15)
    records.extend(save_figure(fig, directory, "difference_arrivals"))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, layout="constrained")
    for row, x in enumerate((175., 345.)):
        index = int(np.flatnonzero(xs == x)[0])
        for c in range(2):
            ax = axes[row, c]
            for difference, pair, color in zip(differences, PAIRS, COLORS, strict=True):
                ax.plot(times, difference[:, index, c]*1e9, label=pair[3], color=color, lw=1.1)
            for pad, color in ((0., "#111111"), (175., "#008444"), (350., "#ad0077")):
                predicted = CENTER+float(path_length(np.array([x]), pad)[0])/speed
                ax.axvline(predicted, color=color, lw=.8, ls="--")
                ax.text(predicted+.004, .96, f"R{pad:g}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=color)
            ax.set(title=f"{'Vx' if c==0 else 'Vy'} · x={x:g} м", ylabel="Разность, нм/с", xlim=(.1, .8))
            ax.grid(alpha=.2)
            if row == 1:
                ax.set_xlabel("Время, с")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Трассы разности · абсолютная линейная шкала в каждой панели", fontsize=15)
    records.extend(save_figure(fig, directory, "arrival_traces"))
    tau, _ = beam(times, xs, differences[0], speed)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    for c in range(2):
        ax = axes[0, c]
        for values, pair, color in zip(beams, PAIRS, COLORS, strict=True):
            ax.plot(tau, values[:, c]*1e9, label=pair[3], color=color, lw=1.3)
        for pad in (0., 175., 350.):
            expected = CENTER+(350+2*pad)/speed
            ax.axvline(expected, color="#555555", ls="--", lw=.7)
            ax.text(expected+.005, .94, f"R{pad:g}", transform=ax.get_xaxis_transform(), va="top")
        ax.set(xlim=(.08, .7), ylabel=f"Среднее {'Vx' if c==0 else 'Vy'}, нм/с", xlabel="τ=t−x/cR, с")
        ax.grid(alpha=.2)
    axes[0, 0].legend(ncols=2, fontsize=9)
    for values, pair, color in zip(beams, PAIRS, COLORS, strict=True):
        amplitude = np.linalg.norm(values, axis=1)*1e9
        axes[1, 0].semilogy(tau, np.maximum(amplitude, 1e-15), label=pair[3], color=color, lw=1.3)
    for pad in (0., 175., 350.):
        expected = CENTER+(350+2*pad)/speed
        axes[1, 0].axvline(expected, color="#555555", ls="--", lw=.7)
        axes[1, 0].text(expected+.005, .96, f"R{pad:g}", transform=axes[1, 0].get_xaxis_transform(), va="top")
    axes[1, 0].set(xlim=(.08, .7), ylim=(1e-8, 1.), ylabel="Модуль среднего разности, нм/с", xlabel="τ=t−x/cR, с")
    axes[1, 0].grid(alpha=.2, which="both")
    for run, label, color in zip(runs, LABELS, COLORS, strict=True):
        rms = np.sqrt(np.mean(run.surface.velocity**2, axis=(1, 2)))
        axes[1, 1].semilogy(times, np.maximum(rms*1e9, 1e-15), label=label, color=color, lw=1.3)
    axes[1, 1].set(xlim=(.25, 2), ylim=(1e-6, 1), xlabel="Физическое время, с", ylabel="RMS поверхности, нм/с")
    axes[1, 1].legend(ncols=2, fontsize=9)
    axes[1, 1].grid(alpha=.2, which="both")
    fig.suptitle("Возврат с левой стороны: среднее после сдвига по скорости Релея\n26 приёмников x=65…315 м; усреднение подавляет часть других наклонов, но не является модовым разложением", fontsize=14)
    records.extend(save_figure(fig, directory, "return_beam_and_tail"))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True, layout="constrained")
    for ax, xmax in zip(axes.ravel(), (175., 225., 275., 315.), strict=True):
        early_peaks: list[float] = []
        at_edges: list[bool] = []
        for difference, label, color in zip(differences[:2], ("175 − 0", "350 − 175"), (COLORS[0], COLORS[2]), strict=True):
            selected_tau, selected_beam = beam(times, xs, difference, speed, xmax)
            amplitude = np.linalg.norm(selected_beam, axis=1)
            full_window = (selected_tau >= .08) & (selected_tau <= .7)
            peak = float(np.max(amplitude[full_window]))
            ax.plot(selected_tau, amplitude/peak, color=color, lw=1.5,
                    label=f"{label} · max={peak*1e9:.4g} нм/с")
            early_window = np.abs(selected_tau-(CENTER+350/speed)) <= .06
            ids = np.flatnonzero(early_window)
            i = int(ids[np.argmax(amplitude[early_window])])
            early_peaks.append(float(selected_tau[i]))
            at_edges.append(bool(i in (ids[0], ids[-1])))
            ax.plot(selected_tau[i], amplitude[i]/peak, "o", color=color, ms=5)
        ax.axvspan(CENTER+350/speed-.06, CENTER+350/speed+.06, color="#dddddd", alpha=.25)
        for padding in (0., 175., 350.):
            expected = CENTER+(350+2*padding)/speed
            ax.axvline(expected, color="#777777", ls=":", lw=.8)
            ax.text(expected+.004, .50, f"R{padding:g}", color="#555555", fontsize=8)
        count = int(np.count_nonzero((xs >= 65) & (xs <= xmax)))
        edge_note = "; максимум на краю окна" if any(at_edges) else ""
        ax.set(title=f"x=65…{xmax:g} м · {count} приёмников\nРанние пики: {early_peaks[0]:.3f} / {early_peaks[1]:.3f} с{edge_note}",
               xlim=(.1, .65), ylim=(0., 1.08), xlabel="τ=t−x/cR, с", ylabel="|Среднее Δv| / собственный максимум")
        ax.grid(alpha=.15)
        ax.legend(loc="upper right", fontsize=9, framealpha=.9)
    fig.suptitle("Проверка времени прихода при разных апертурах\nКаждая кривая нормирована на свой максимум в τ=0,08…0,70 с — только сравнение времени, не амплитуд\nСерое окно R0±60 мс; точки — максимум внутри окна; R175−R0=157,24 мс", fontsize=13)
    records.extend(save_figure(fig, directory, "incoming_aperture_check"))
    alpha_max = number(mapping(runs[1].metadata["run_spec"])["alpha_max_s_inv"])
    distance = np.linspace(0., 350., 701)
    original = np.where(distance <= 175., alpha_max*(distance/175.)**2, np.nan)
    stretched = alpha_max*(distance/350.)**2
    held = alpha_max*np.minimum(distance/175., 1.)**2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for values, label, color, style in zip((original, stretched, held), LABELS[1:], COLORS[1:], ("-", "-", "--"), strict=True):
        axes[0].plot(distance, values, color=color, label=label, lw=2., ls=style)
    for location, label in ((175., "ABC 175"), (350., "ABC 350")):
        axes[0].axvline(location, color="#555555", ls=":", lw=1.)
        axes[0].text(location-4., .98, label, transform=axes[0].get_xaxis_transform(), ha="right", va="top", fontsize=10)
    axes[0].set(xlim=(0., 355.), ylim=(0., alpha_max*1.13), xlabel="Расстояние от входа в слой s, м", ylabel="α(s), с⁻¹", title=f"Одинаковое αmax={alpha_max:.3f} с⁻¹")
    axes[0].grid(alpha=.2)
    axes[0].legend(loc="upper left", fontsize=9)
    bars = axes[1].bar([0, 1, 2], [1., 2., 4.], color=COLORS[1:], width=.6)
    for bar, value in zip(bars, (1., 2., 4.), strict=True):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.08, f"{value:g}×", ha="center", fontsize=14)
    axes[1].set(xticks=[0, 1, 2], xticklabels=["175 м", "350 м\nплавный", "350 м\n175 + плато"], ylim=(0., 4.7), ylabel="Интеграл α(s) ds относительно слоя175", title="Интеграл вдоль одной нормали")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Что изменяется при удвоении слоя\nC=αM; в нижних углах вклады двух направлений складываются · интеграл не равен коэффициенту отражения", fontsize=14)
    records.extend(save_figure(fig, directory, "damping_profiles"))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--double", type=Path, default=DATA / "double_layer_3060_v1/padded350")
    parser.add_argument("--held", type=Path, default=DATA / "double_layer_3060_v1/held175_profile")
    parser.add_argument("--out", type=Path, default=DATA / "damping_layer_analysis_v1")
    parser.add_argument("--images", type=Path, default=IMAGES / "damping_layer_comparison_v1")
    args = parser.parse_args()
    output = cast(Path, args.out).resolve()
    images = cast(Path, args.images).resolve()
    output.relative_to(DATA)
    images.relative_to(IMAGES)
    if output.exists() or images.exists():
        raise FileExistsError("Use a new analysis/figure directory")
    paths = [DATA / "homogeneous_3060_v1/core", DATA / "homogeneous_3060_v1/padded", args.double, args.held]
    runs = [load_run(name, path) for name, path in zip(NAMES, paths, strict=True)]
    for run in runs[1:]:
        validate_pair(runs[0].metadata, run.metadata)
        if not np.array_equal(runs[0].surface.times, run.surface.times) or not np.allclose(runs[0].surface.coordinates, run.surface.coordinates, atol=1e-9, rtol=0):
            raise ValueError("Receiver/time axes differ")
    for run, expected in zip(runs[2:], ("stretched350", "fixed175plateau"), strict=True):
        if run.metadata.get("padding_m") != 350. or mapping(run.metadata["damping_profile"])["name"] != expected:
            raise ValueError("Incorrect double-domain control")
    result, differences, beams = diagnostics(runs)
    output.mkdir(parents=True, exist_ok=False)
    images.mkdir(parents=True, exist_ok=False)
    result.update(status="completed", utc=datetime.now(timezone.utc).isoformat(),
        schema="chapter14.damping_layer_analysis.v1", inputs={run.name: {"metadata": file_evidence(run.path/"metadata.json"), "surface": file_evidence(run.path/"surface.npz")} for run in runs},
        code=file_evidence(Path(__file__)), figures=figures(runs, differences, beams, images),
        method={"rayleigh_equation": "(2-q^2)^2 = 4 sqrt(1-q^2) sqrt(1-(Vs/Vp)^2 q^2), q=cR/Vs; positive nonzero subsonic root", "rayleigh_derivation": "Plane P/S potentials with exp(-a k z), exp(-b k z); zero tangential/normal traction gives determinant above. Used only as a kinematic guide in the undamped homogeneous medium.",
            "ricker_center_s": CENTER, "onset": "First 3 consecutive 1ms samples exceeding each stated amplitude threshold; Ricker is not compact in time, so threshold onsets are not exact wavefronts", "reference_window_peak": "Single175 velocity maximum in [0,t0+abs(x-350)/cR+60ms]; a reproducible reference window, not a pure direct-mode separation. Near x=5 direct and boundary-related arrivals can overlap.", "receiver_gate_caution": "after_direct_plus80ms intentionally excludes the initial wave; at near-edge receivers this can also exclude early returning waves, so it is not a measure of all reflected response", "late_amplitude_caution": "RMS and squared velocity sum are amplitude proxies, not elastic energy or true numerical error; no infinite-domain reference was computed", "differences_caution": "Differences between approximate boundary treatments are not true error. Both damping ramp and boundary position change in stretched350; held175plateau isolates moving the old boundary while retaining the old interior coefficients."})
    with (output/"analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "out": str(output), "causality": result["held_profile_causality"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
