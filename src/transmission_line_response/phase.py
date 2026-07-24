from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocess import PreprocessConfig, prepare_s21
from .s21 import group_delay_from_s21
from .types import PreparedS21, RawS21Data


@dataclass(frozen=True)
class PhaseAnalysisConfig:
    fit_points: int = 100
    trusted_start_ghz: float = 0.0
    trusted_stop_ghz: float = 1.0
    magnitude_floor_db: float = -45.0
    smoothing_points: int = 21
    deviation_limit_ns: float = 0.5


@dataclass(frozen=True)
class PhaseGroupDelayResult:
    prepared: PreparedS21
    config: PhaseAnalysisConfig
    phase_slope_rad_per_ghz: float
    phase_intercept_rad: float
    delay_ns: float
    fitted_phase_rad: np.ndarray
    compensated_phase_rad: np.ndarray
    group_delay_ns: np.ndarray
    trusted_start_ghz: float
    trusted_stop_ghz: float
    trusted_gd_mean_ns: float
    trusted_gd_std_ns: float
    trusted_gd_pp_ns: float


def fit_linear_phase(freq_ghz: np.ndarray, phase_rad: np.ndarray, points: int = 100) -> tuple[float, float, float]:
    count = min(max(2, int(points)), len(freq_ghz))
    slope, intercept = np.polyfit(freq_ghz[:count], phase_rad[:count], 1)
    delay_ns = float(-slope / (2.0 * np.pi))
    return float(slope), float(intercept), delay_ns


def compensate_linear_phase(freq_ghz: np.ndarray, phase_rad: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return phase_rad - (float(slope) * freq_ghz + float(intercept))


def select_trusted_group_delay_band(
    freq_ghz: np.ndarray,
    s21_db: np.ndarray,
    group_delay_ns: np.ndarray,
    config: PhaseAnalysisConfig | None = None,
) -> tuple[float, float]:
    cfg = config or PhaseAnalysisConfig()
    window = max(1, int(cfg.smoothing_points))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    smooth = np.convolve(group_delay_ns, kernel, mode="same")
    deviation = np.abs(group_delay_ns - smooth)
    mask = (
        (freq_ghz >= cfg.trusted_start_ghz)
        & (freq_ghz <= cfg.trusted_stop_ghz)
        & (s21_db >= cfg.magnitude_floor_db)
        & (deviation <= cfg.deviation_limit_ns)
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return float(cfg.trusted_start_ghz), float(cfg.trusted_stop_ghz)
    start = int(idx[0])
    end = start
    while end + 1 < mask.size and mask[end + 1]:
        end += 1
    return float(freq_ghz[start]), float(freq_ghz[end])


def analyze_phase_group_delay(
    source: str | Path | RawS21Data | PreparedS21,
    *,
    preprocess_config: PreprocessConfig | None = None,
    phase_config: PhaseAnalysisConfig | None = None,
    label: str | None = None,
) -> PhaseGroupDelayResult:
    prepared = source if isinstance(source, PreparedS21) else prepare_s21(source, preprocess_config, label=label)
    cfg = phase_config or PhaseAnalysisConfig()
    phase = np.unwrap(prepared.phase_rad)
    slope, intercept, delay_ns = fit_linear_phase(prepared.freq_ghz, phase, cfg.fit_points)
    fitted = slope * prepared.freq_ghz + intercept
    compensated = phase - fitted
    gd_ns = group_delay_from_s21(prepared.s21, freqs=prepared.freq_ghz)
    trusted_start, trusted_stop = select_trusted_group_delay_band(prepared.freq_ghz, prepared.s21_db, gd_ns, cfg)
    trusted_mask = (prepared.freq_ghz >= trusted_start) & (prepared.freq_ghz <= trusted_stop)
    trusted = gd_ns[trusted_mask] if np.any(trusted_mask) else gd_ns
    return PhaseGroupDelayResult(
        prepared=prepared,
        config=cfg,
        phase_slope_rad_per_ghz=slope,
        phase_intercept_rad=intercept,
        delay_ns=delay_ns,
        fitted_phase_rad=fitted,
        compensated_phase_rad=compensated,
        group_delay_ns=gd_ns,
        trusted_start_ghz=trusted_start,
        trusted_stop_ghz=trusted_stop,
        trusted_gd_mean_ns=float(np.mean(trusted)),
        trusted_gd_std_ns=float(np.std(trusted)),
        trusted_gd_pp_ns=float(np.max(trusted) - np.min(trusted)),
    )
