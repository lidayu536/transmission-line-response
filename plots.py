from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np

from .phase import PhaseGroupDelayResult
from .time_domain import TimeDomainResponse, first_level_crossing


def plot_rise_analysis(
    output_path: str | Path,
    response: TimeDomainResponse,
    *,
    title: str | None = None,
    xlim_ns: tuple[float, float] = (-5.0, 40.0),
    mark_levels: Sequence[float] = (0.10, 0.90, 0.95),
) -> Path:
    path = Path(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.6), dpi=220)
    axes = np.atleast_1d(axes)
    prepared = response.prepared

    axes[0].plot(prepared.freq_ghz, prepared.s21_db, label=f"{prepared.label}: equivalent |S21|")
    axes[0].set_xlim(float(prepared.freq_ghz[0]), float(prepared.freq_ghz[-1]))
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Equivalent |S21| (dB)")
    axes[0].set_title("S21 magnitude after preprocessing")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(response.aligned_time_ns, np.real(response.impulse), label=f"{prepared.label}: impulse response")
    axes[1].set_xlim(xlim_ns)
    axes[1].set_xlabel("Aligned time (ns)")
    axes[1].set_ylabel("Impulse response")
    axes[1].set_title("Impulse response")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(response.aligned_time_ns, response.step_normalized, label=f"{prepared.label}: normalized step response")
    colors = {0.10: "tab:orange", 0.90: "tab:green", 0.95: "tab:red"}
    for level in mark_levels:
        crossing = first_level_crossing(response.aligned_time_ns, response.step_normalized, float(level), after_x=0.0)
        if crossing is None:
            continue
        color = colors.get(round(float(level), 2), "tab:purple")
        axes[2].scatter([crossing], [level], color=color, s=24, zorder=3)
        axes[2].annotate(f"{level:.0%} @ {crossing:.2f} ns", (crossing + 0.5, level - 0.04), fontsize=8, color=color)
    axes[2].set_xlim(xlim_ns)
    axes[2].set_ylim(-0.08, 1.12)
    axes[2].set_xlabel("Aligned time (ns)")
    axes[2].set_ylabel("Normalized step")
    axes[2].set_title("Step response and rise-edge markers")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _aligned_time_by_level(response: TimeDomainResponse, level: float) -> tuple[np.ndarray, float | None]:
    crossing = first_level_crossing(response.aligned_time_ns, response.step_normalized, level, after_x=0.0)
    if crossing is None:
        return response.aligned_time_ns.copy(), None
    return response.aligned_time_ns - crossing, crossing


def plot_rise_comparison(
    output_path: str | Path,
    responses: Sequence[TimeDomainResponse],
    *,
    title: str | None = None,
    align_level: float = 0.01,
    xlim_ns: tuple[float, float] = (-2.0, 35.0),
) -> Path:
    path = Path(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.8), dpi=220)
    axes = np.atleast_1d(axes)

    for response in responses:
        label = response.prepared.label
        axes[0].plot(response.prepared.freq_ghz, response.prepared.s21_db, label=f"{label}: equivalent |S21|")
        axes[1].plot(response.aligned_time_ns, np.real(response.impulse), label=f"{label}: impulse")
        shifted_time, start = _aligned_time_by_level(response, align_level)
        axes[2].plot(shifted_time, response.step_normalized, label=f"{label}: step aligned at {align_level:.0%}")
        t95 = response.metrics.t95_ns
        if start is not None and t95 is not None:
            axes[2].scatter([t95 - start], [0.95], s=26, zorder=3)
            axes[2].annotate(f"95% @ {t95 - start:.2f} ns", (t95 - start + 0.45, 0.90), fontsize=8)

    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Equivalent |S21| (dB)")
    axes[0].set_title("Single-pass equivalent S21 magnitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlim(xlim_ns)
    axes[1].set_xlabel("Aligned time (ns)")
    axes[1].set_ylabel("Impulse response")
    axes[1].set_title("Impulse response")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_xlim(xlim_ns)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_xlabel(f"Time after {align_level:.0%} crossing (ns)")
    axes[2].set_ylabel("Normalized step")
    axes[2].set_title("Rise-edge comparison")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_phase_group_delay(
    output_path: str | Path,
    result: PhaseGroupDelayResult,
    *,
    title: str | None = None,
    max_freq_ghz: float = 1.0,
) -> Path:
    path = Path(output_path)
    mask = result.prepared.freq_ghz <= max_freq_ghz
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), dpi=220)
    axes = np.atleast_1d(axes)

    axes[0].plot(result.prepared.freq_ghz[mask], result.prepared.phase_rad[mask], label="single-pass equivalent phase")
    axes[0].plot(result.prepared.freq_ghz[mask], result.fitted_phase_rad[mask], linestyle="--", label="linear phase fit")
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Phase (rad)")
    axes[0].set_title("Unwrapped phase")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(result.prepared.freq_ghz[mask], result.compensated_phase_rad[mask], label="residual phase after linear compensation")
    axes[1].axhline(0.0, color="tab:red", linestyle=":", linewidth=0.9, label="zero phase reference")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("Residual phase (rad)")
    axes[1].set_title("Compensated residual phase")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(result.prepared.freq_ghz[mask], result.group_delay_ns[mask], label="group delay from unwrapped phase")
    axes[2].axvspan(result.trusted_start_ghz, result.trusted_stop_ghz, color="tab:green", alpha=0.12, label="trusted low-frequency interval")
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].set_ylabel("Group delay (ns)")
    axes[2].set_title("Group delay")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
