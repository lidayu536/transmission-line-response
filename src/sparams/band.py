from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.arrays import ArrayLike1D, _as_1d_array


@dataclass(frozen=True)
class S21BandwidthWindow:
    freqs: np.ndarray
    s21: np.ndarray
    magnitude_db: np.ndarray
    smoothed_magnitude_db: np.ndarray
    threshold_db: float
    cutoff_index: int
    cutoff_frequency_hz: float
    taper_points: int
    was_truncated: bool


def select_usable_s21_band(
    freqs: ArrayLike1D,
    s21: ArrayLike1D,
    *,
    relative_floor_db: float = 60.0,
    absolute_floor_db: float = -80.0,
    smoothing_points: int = 17,
    sustain_points: int = 25,
    taper_fraction: float = 0.125,
    min_points: int = 64,
) -> S21BandwidthWindow:
    freqs_array = _as_1d_array(freqs, name="freqs", dtype=float)
    s21_array = _as_1d_array(s21, name="s21", dtype=complex)
    if freqs_array.shape != s21_array.shape:
        raise ValueError("freqs and s21 must have the same shape.")
    if freqs_array.size < 2:
        raise ValueError("Need at least two frequency points to select a usable S21 band.")

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(s21_array), 1e-300))
    threshold_db = max(float(magnitude_db[0] - relative_floor_db), float(absolute_floor_db))

    smoothing_points = max(1, int(smoothing_points))
    if smoothing_points % 2 == 0:
        smoothing_points += 1
    kernel = np.ones(smoothing_points, dtype=float) / smoothing_points
    smoothed = np.convolve(magnitude_db, kernel, mode="same")

    sustain_points = max(1, int(sustain_points))
    min_points = max(8, int(min_points))
    cutoff_index = freqs_array.size
    upper_bound = freqs_array.size - sustain_points + 1
    for start in range(min_points, upper_bound):
        if np.all(smoothed[start : start + sustain_points] <= threshold_db):
            cutoff_index = start
            break

    cutoff_index = min(max(cutoff_index, min_points), freqs_array.size)
    truncated = cutoff_index < freqs_array.size
    freqs_cut = freqs_array[:cutoff_index].copy()
    s21_cut = s21_array[:cutoff_index].copy()

    taper_points = 0
    if truncated and taper_fraction > 0.0 and cutoff_index >= 8:
        taper_points = max(4, int(round(cutoff_index * taper_fraction)))
        taper_points = min(taper_points, cutoff_index // 2)
        if taper_points > 0:
            taper = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper_points)))
            s21_cut[-taper_points:] *= taper

    return S21BandwidthWindow(
        freqs=freqs_cut,
        s21=s21_cut,
        magnitude_db=magnitude_db,
        smoothed_magnitude_db=smoothed,
        threshold_db=threshold_db,
        cutoff_index=cutoff_index,
        cutoff_frequency_hz=float(freqs_cut[-1]),
        taper_points=taper_points,
        was_truncated=truncated,
    )
