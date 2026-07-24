from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import read_s21
from ..core import PreparedS21, RawS21Data


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for converting raw S21 samples onto a working grid."""

    grid_start_hz: float = 0.0
    grid_stop_hz: float = 5.0e9
    grid_step_hz: float = 5.0e6
    include_grid_stop: bool = False
    skip_initial_points: int = 0
    add_dc_if_needed: bool = True
    dc_phase_rad: float = 0.0
    dc_magnitude_method: str = "first_two_db"
    dc_fit_points: int = 5
    roundtrip_to_single_pass: bool = False


def repo_roundtrip_s2p_config(
    *,
    grid_stop_hz: float = 5.0e9,
    grid_step_hz: float = 5.0e6,
    skip_initial_points: int = 1,
) -> PreprocessConfig:
    """Preset matching the notebook-style repo S2P workflow."""

    return PreprocessConfig(
        grid_stop_hz=grid_stop_hz,
        grid_step_hz=grid_step_hz,
        skip_initial_points=skip_initial_points,
        add_dc_if_needed=True,
        dc_phase_rad=0.0,
        dc_magnitude_method="first_two_db",
        roundtrip_to_single_pass=True,
    )


def check_fft_ready_grid(
    freq_hz: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol_hz: float = 1e-6,
) -> tuple[bool, float | None]:
    """Check whether `freq_hz` is close to `df * arange(n)`."""

    freq = np.asarray(freq_hz, dtype=float).reshape(-1)
    if freq.size < 2:
        return False, None
    df = float(freq[1] - freq[0])
    if df <= 0.0:
        return False, None
    expected = df * np.arange(freq.size, dtype=float)
    ready = bool(np.isclose(freq[0], 0.0, atol=atol_hz) and np.allclose(freq, expected, rtol=rtol, atol=atol_hz))
    return ready, df if ready else None


def _dc_magnitude_db(freq_hz: np.ndarray, mag_db: np.ndarray, config: PreprocessConfig) -> float:
    method = config.dc_magnitude_method.lower()
    if mag_db.size == 1 or method == "hold":
        return float(mag_db[0])
    if method == "first_two_db":
        return float(2.0 * mag_db[0] - mag_db[1])
    if method == "fit_lowfreq_db":
        points = min(max(2, config.dc_fit_points), mag_db.size)
        slope, intercept = np.polyfit(freq_hz[:points], mag_db[:points], 1)
        return float(intercept)
    raise ValueError(f"Unsupported dc_magnitude_method: {config.dc_magnitude_method}")


def add_dc_anchor(
    freq_hz: np.ndarray,
    mag_db: np.ndarray,
    phase_rad: np.ndarray,
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, bool]:
    """Add a 0 Hz anchor using dB extrapolation and `phase0` when needed."""

    freq = np.asarray(freq_hz, dtype=float).reshape(-1)
    mag = np.asarray(mag_db, dtype=float).reshape(-1)
    phase = np.asarray(phase_rad, dtype=float).reshape(-1)
    if not config.add_dc_if_needed or (freq.size > 0 and np.isclose(freq[0], 0.0)):
        return freq, mag, phase, None, False
    anchor_db = _dc_magnitude_db(freq, mag, config)
    return (
        np.concatenate([[0.0], freq]),
        np.concatenate([[anchor_db], mag]),
        np.concatenate([[float(config.dc_phase_rad)], phase]),
        anchor_db,
        True,
    )


def _working_grid(config: PreprocessConfig) -> np.ndarray:
    stop = config.grid_stop_hz + (0.5 * config.grid_step_hz if config.include_grid_stop else 0.0)
    grid = np.arange(config.grid_start_hz, stop, config.grid_step_hz, dtype=float)
    if grid.size < 2:
        raise ValueError("Working frequency grid must contain at least two points.")
    return grid


def prepare_s21(
    source: str | Path | RawS21Data,
    config: PreprocessConfig | None = None,
    *,
    label: str | None = None,
) -> PreparedS21:
    """Read and preprocess S21 onto a zero-start uniform grid.

    The phase is unwrapped immediately after reading raw samples. The 0 Hz anchor
    and interpolation are then applied to dB and unwrapped phase separately.
    """

    cfg = config or PreprocessConfig()
    raw = read_s21(source) if isinstance(source, (str, Path)) else source
    notes: list[str] = []

    start = max(0, int(cfg.skip_initial_points))
    if start >= raw.freq_hz.size - 1:
        raise ValueError("skip_initial_points removes too many samples.")
    if start:
        notes.append(f"Skipped the first {start} raw sample(s).")

    freq = raw.freq_hz[start:].astype(float)
    s21 = raw.s21[start:].astype(complex)
    raw_ready, raw_df = check_fft_ready_grid(freq)
    notes.append(f"Raw grid FFT-ready: {raw_ready}" + (f", df={raw_df:.6g} Hz." if raw_df else "."))

    mag_db = 20.0 * np.log10(np.maximum(np.abs(s21), 1e-300))
    phase_rad = np.unwrap(np.angle(s21))
    freq_anchor, mag_anchor, phase_anchor, dc_db, dc_added = add_dc_anchor(freq, mag_db, phase_rad, cfg)
    if dc_added:
        notes.append(f"Added 0 Hz anchor: magnitude={dc_db:.6g} dB, phase={cfg.dc_phase_rad:.6g} rad.")

    grid_hz = _working_grid(cfg)
    if grid_hz[0] < freq_anchor[0] or grid_hz[-1] > freq_anchor[-1]:
        notes.append("Working grid extends outside raw frequency coverage; edge values use numpy.interp endpoint hold.")
    grid_db = np.interp(grid_hz, freq_anchor, mag_anchor)
    grid_phase = np.interp(grid_hz, freq_anchor, phase_anchor)
    fullpass_s21 = np.power(10.0, grid_db / 20.0) * np.exp(1j * grid_phase)

    if cfg.roundtrip_to_single_pass:
        s21_db = 0.5 * grid_db
        phase = 0.5 * grid_phase
        notes.append("Constructed single-pass equivalent by halving dB and phase.")
    else:
        s21_db = grid_db
        phase = grid_phase
        notes.append("Used S21 directly without round-trip halving.")
    s21_equiv = np.power(10.0, s21_db / 20.0) * np.exp(1j * phase)

    return PreparedS21(
        label=label or raw.label,
        raw=raw,
        freq_hz=grid_hz,
        freq_ghz=grid_hz / 1.0e9,
        fullpass_db=grid_db,
        fullpass_phase_rad=grid_phase,
        s21_db=s21_db,
        phase_rad=phase,
        s21=s21_equiv,
        single_pass_assumed=cfg.roundtrip_to_single_pass,
        dc_added=dc_added,
        dc_magnitude_db=dc_db,
        grid_step_hz=float(cfg.grid_step_hz),
        notes=tuple(notes),
    )
