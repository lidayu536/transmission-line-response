from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sparams.preprocess import PreprocessConfig, prepare_s21
from ..sparams.transforms import s21_to_impulse_response
from ..core import PreparedS21, RawS21Data


@dataclass(frozen=True)
class IfftConfig:
    phase_fit_points: int = 100
    interp_multiple: int | None = None
    extend_multiple: int | None = None
    linear_extend: bool | None = None
    linear_extend_ref: int = 200
    target_dt_ns: float = 0.05
    min_time_span_ns: float = 100.0
    max_interp_multiple: int = 20
    max_extend_multiple: int = 10
    align_offset_ns: float = 0.0
    normalization_mode: str = "reference_point"
    normalization_time_ns: float = 200.0
    baseline_window_ns: tuple[float, float] = (-60.0, -20.0)
    plateau_window_ns: tuple[float, float] = (80.0, 120.0)


@dataclass(frozen=True)
class IfftParameterChoice:
    interp_multiple: int
    extend_multiple: int
    linear_extend: bool
    linear_extend_ref: int
    edge_s21_db: float
    base_span_ns: float
    native_dt_ns: float
    rationale: str


@dataclass(frozen=True)
class RiseMetrics:
    t01_ns: float | None
    t10_ns: float | None
    t50_ns: float | None
    t90_ns: float | None
    t95_ns: float | None
    rise_10_90_ns: float | None
    overshoot: float
    plateau_std: float
    plateau_pp: float


@dataclass(frozen=True)
class TimeDomainResponse:
    prepared: PreparedS21
    config: IfftConfig
    ifft_choice: IfftParameterChoice
    phase_delay_ns: float
    align_reference_ns: float
    time_ns: np.ndarray
    aligned_time_ns: np.ndarray
    impulse: np.ndarray
    step_raw: np.ndarray
    step_normalized: np.ndarray
    baseline_level: float
    plateau_level: float
    metrics: RiseMetrics


def first_level_crossing(
    x: np.ndarray,
    y: np.ndarray,
    level: float,
    *,
    after_x: float = 0.0,
) -> float | None:
    x_work = np.asarray(x, dtype=float)
    y_work = np.asarray(y, dtype=float)
    mask = x_work >= after_x
    x_work = x_work[mask]
    y_work = y_work[mask]
    if x_work.size < 2:
        return None
    idxs = np.flatnonzero((y_work[:-1] <= level) & (y_work[1:] >= level))
    if idxs.size == 0:
        return None
    idx = int(idxs[0])
    x0 = float(x_work[idx])
    x1 = float(x_work[idx + 1])
    y0 = float(y_work[idx])
    y1 = float(y_work[idx + 1])
    if np.isclose(y1, y0):
        return x0
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def _fit_phase_delay_ns(prepared: PreparedS21, points: int) -> float:
    count = min(max(2, int(points)), prepared.freq_ghz.size)
    slope, _ = np.polyfit(prepared.freq_ghz[:count], np.unwrap(np.angle(prepared.s21[:count])), 1)
    return float(-slope / (2.0 * np.pi))


def choose_ifft_parameters(prepared: PreparedS21, config: IfftConfig | None = None) -> IfftParameterChoice:
    cfg = config or IfftConfig()
    df_ghz = float(prepared.freq_ghz[1] - prepared.freq_ghz[0])
    df_hz = df_ghz * 1.0e9
    max_freq_hz = float(prepared.freq_hz[-1])
    base_span_ns = 1.0e9 / df_hz
    native_dt_ns = 1.0e9 / (2.0 * max_freq_hz) if max_freq_hz > 0.0 else cfg.target_dt_ns
    edge_s21_db = float(20.0 * np.log10(np.maximum(np.abs(prepared.s21[-1]), 1e-300)))
    align_reference_ns = _fit_phase_delay_ns(prepared, cfg.phase_fit_points) + float(cfg.align_offset_ns)
    if cfg.normalization_mode == "reference_point":
        latest_needed_after_align_ns = cfg.normalization_time_ns
    elif cfg.normalization_mode == "window":
        latest_needed_after_align_ns = max(abs(cfg.baseline_window_ns[0]), abs(cfg.baseline_window_ns[1]), cfg.plateau_window_ns[1])
    else:
        latest_needed_after_align_ns = cfg.min_time_span_ns / 2.0
    needed_total_span_ns = 2.0 * (abs(align_reference_ns) + latest_needed_after_align_ns + 20.0)
    target_span_ns = max(cfg.min_time_span_ns, needed_total_span_ns)

    if cfg.extend_multiple is None:
        extend_multiple = max(1, int(np.ceil(native_dt_ns / cfg.target_dt_ns)))
        extend_multiple = min(extend_multiple, cfg.max_extend_multiple)
    else:
        extend_multiple = int(cfg.extend_multiple)

    if cfg.interp_multiple is None:
        interp_multiple = max(1, int(np.ceil(target_span_ns / base_span_ns)))
        interp_multiple = min(interp_multiple, cfg.max_interp_multiple)
    else:
        interp_multiple = int(cfg.interp_multiple)

    if cfg.linear_extend is None:
        linear_extend = bool(edge_s21_db > -50.0)
    else:
        linear_extend = bool(cfg.linear_extend)

    rationale = (
        f"df={df_hz:.6g} Hz gives a base window of {base_span_ns:.3f} ns; "
        f"the estimated aligned reference is {align_reference_ns:.3f} ns and the requested usable span is {target_span_ns:.3f} ns; "
        f"band-edge magnitude is {edge_s21_db:.3f} dB; selected "
        f"interp_multiple={interp_multiple}, extend_multiple={extend_multiple}, "
        f"linear_extend={linear_extend}."
    )
    return IfftParameterChoice(
        interp_multiple=interp_multiple,
        extend_multiple=extend_multiple,
        linear_extend=linear_extend,
        linear_extend_ref=int(cfg.linear_extend_ref),
        edge_s21_db=edge_s21_db,
        base_span_ns=float(base_span_ns),
        native_dt_ns=float(native_dt_ns),
        rationale=rationale,
    )


def _mean_in_window(x: np.ndarray, y: np.ndarray, window: tuple[float, float], fallback: float) -> float:
    mask = (x >= window[0]) & (x <= window[1])
    if np.any(mask):
        return float(np.mean(y[mask]))
    return fallback


def _normalize_step(
    aligned_time_ns: np.ndarray,
    step_raw: np.ndarray,
    align_index: int,
    config: IfftConfig,
) -> tuple[np.ndarray, float, float]:
    if config.normalization_mode == "window":
        baseline = _mean_in_window(aligned_time_ns, step_raw, config.baseline_window_ns, float(step_raw[align_index]))
        plateau = _mean_in_window(aligned_time_ns, step_raw, config.plateau_window_ns, float(step_raw[-1]))
    elif config.normalization_mode == "full_scale":
        baseline = float(np.min(step_raw))
        plateau = float(np.max(step_raw))
    elif config.normalization_mode == "reference_point":
        baseline = float(step_raw[align_index])
        dt = float(aligned_time_ns[1] - aligned_time_ns[0])
        plateau_index = align_index + int(round(config.normalization_time_ns / dt))
        plateau_index = min(max(0, plateau_index), step_raw.size - 1)
        plateau = float(step_raw[plateau_index])
    else:
        raise ValueError(f"Unsupported normalization_mode: {config.normalization_mode}")
    denom = plateau - baseline
    if np.isclose(denom, 0.0):
        denom = 1.0
    return (step_raw - baseline) / denom, baseline, plateau


def _measure_rise_metrics(
    aligned_time_ns: np.ndarray,
    step_normalized: np.ndarray,
    config: IfftConfig,
) -> RiseMetrics:
    t01 = first_level_crossing(aligned_time_ns, step_normalized, 0.01, after_x=0.0)
    t10 = first_level_crossing(aligned_time_ns, step_normalized, 0.10, after_x=0.0)
    t50 = first_level_crossing(aligned_time_ns, step_normalized, 0.50, after_x=0.0)
    t90 = first_level_crossing(aligned_time_ns, step_normalized, 0.90, after_x=0.0)
    t95 = first_level_crossing(aligned_time_ns, step_normalized, 0.95, after_x=0.0)
    rise = None if t10 is None or t90 is None else float(t90 - t10)
    plateau_mask = (aligned_time_ns >= config.plateau_window_ns[0]) & (aligned_time_ns <= config.plateau_window_ns[1])
    plateau = step_normalized[plateau_mask] if np.any(plateau_mask) else step_normalized
    max_after = step_normalized[aligned_time_ns >= 0.0]
    overshoot = float(np.max(max_after) - 1.0) if max_after.size else float("nan")
    return RiseMetrics(
        t01_ns=t01,
        t10_ns=t10,
        t50_ns=t50,
        t90_ns=t90,
        t95_ns=t95,
        rise_10_90_ns=rise,
        overshoot=overshoot,
        plateau_std=float(np.std(plateau)),
        plateau_pp=float(np.max(plateau) - np.min(plateau)),
    )


def recover_time_domain_response(
    prepared: PreparedS21,
    config: IfftConfig | None = None,
) -> TimeDomainResponse:
    cfg = config or IfftConfig()
    choice = choose_ifft_parameters(prepared, cfg)
    phase_delay_ns = _fit_phase_delay_ns(prepared, cfg.phase_fit_points)
    align_reference_ns = phase_delay_ns + float(cfg.align_offset_ns)
    time_ns, impulse = s21_to_impulse_response(
        prepared.s21,
        prepared.freq_ghz[1] - prepared.freq_ghz[0],
        extend_multiple=choice.extend_multiple,
        inter_multiple=choice.interp_multiple,
        is_linear_extend=choice.linear_extend,
        linear_extend_ref=choice.linear_extend_ref,
    )
    dt_ns = float(time_ns[1] - time_ns[0])
    step_raw = np.cumsum(np.real(impulse)) * dt_ns
    align_index = int(np.argmin(np.abs(time_ns - align_reference_ns)))
    aligned_time_ns = time_ns - time_ns[align_index]
    step_normalized, baseline, plateau = _normalize_step(aligned_time_ns, step_raw, align_index, cfg)
    metrics = _measure_rise_metrics(aligned_time_ns, step_normalized, cfg)
    return TimeDomainResponse(
        prepared=prepared,
        config=cfg,
        ifft_choice=choice,
        phase_delay_ns=phase_delay_ns,
        align_reference_ns=align_reference_ns,
        time_ns=time_ns,
        aligned_time_ns=aligned_time_ns,
        impulse=impulse,
        step_raw=step_raw,
        step_normalized=step_normalized,
        baseline_level=baseline,
        plateau_level=plateau,
        metrics=metrics,
    )


def analyze_rise_from_s21(
    source: str | Path | RawS21Data,
    *,
    preprocess_config: PreprocessConfig | None = None,
    ifft_config: IfftConfig | None = None,
    label: str | None = None,
) -> TimeDomainResponse:
    prepared = prepare_s21(source, preprocess_config, label=label)
    return recover_time_domain_response(prepared, ifft_config)
