from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizedStepTrace:
    time_seconds: np.ndarray
    normalized_signal: np.ndarray
    edge_index: int
    edge_time_seconds: float
    baseline: float
    plateau: float
    direction: int


def find_step_edge_index(
    signal: np.ndarray,
    *,
    threshold_ratio: float = 0.8,
    prefer_earliest: bool = True,
) -> int:
    signal_array = np.asarray(signal, dtype=float)
    if signal_array.ndim != 1 or signal_array.size < 2:
        raise ValueError("signal must be a 1D array with at least two points.")

    diffs = np.diff(signal_array)
    max_abs = float(np.max(np.abs(diffs)))
    if np.isclose(max_abs, 0.0):
        return int(signal_array.size // 2)

    if prefer_earliest:
        candidates = np.flatnonzero(np.abs(diffs) >= threshold_ratio * max_abs)
        if candidates.size > 0:
            return int(candidates[0])
    return int(np.argmax(np.abs(diffs)))


def normalize_step_trace(
    time_seconds: np.ndarray,
    signal: np.ndarray,
    *,
    edge_index: int | None = None,
    edge_time_seconds: float | None = None,
    threshold_ratio: float = 0.8,
    prefer_earliest: bool = True,
    pre_window_points: int | None = None,
    post_window_points: int | None = None,
) -> NormalizedStepTrace:
    time_array = np.asarray(time_seconds, dtype=float)
    signal_array = np.asarray(signal, dtype=float)
    if time_array.ndim != 1 or signal_array.ndim != 1 or time_array.shape != signal_array.shape:
        raise ValueError("time_seconds and signal must be 1D arrays with the same shape.")

    n = time_array.size
    if edge_time_seconds is not None:
        edge_index = int(np.argmin(np.abs(time_array - edge_time_seconds)))
    elif edge_index is None:
        edge_index = find_step_edge_index(
            signal_array,
            threshold_ratio=threshold_ratio,
            prefer_earliest=prefer_earliest,
        )

    if pre_window_points is None:
        pre_window_points = max(64, n // 40)
    if post_window_points is None:
        post_window_points = max(64, n // 20)

    pre_end = max(1, edge_index)
    pre_start = max(0, pre_end - pre_window_points)
    post_start = min(n - 1, edge_index + 1)
    post_end = min(n, post_start + post_window_points)

    if pre_end <= pre_start:
        pre_start, pre_end = 0, max(1, min(n // 20, n))
    if post_end <= post_start:
        post_start, post_end = max(0, n - max(1, n // 20)), n

    baseline = float(np.mean(signal_array[pre_start:pre_end]))
    plateau = float(np.mean(signal_array[post_start:post_end]))
    direction = 1 if plateau >= baseline else -1
    denominator = plateau - baseline
    if np.isclose(denominator, 0.0):
        denominator = 1.0

    if direction > 0:
        normalized = (signal_array - baseline) / denominator
    else:
        normalized = (baseline - signal_array) / (baseline - plateau if not np.isclose(baseline - plateau, 0.0) else 1.0)

    return NormalizedStepTrace(
        time_seconds=time_array - time_array[edge_index],
        normalized_signal=normalized,
        edge_index=edge_index,
        edge_time_seconds=float(time_array[edge_index]),
        baseline=baseline,
        plateau=plateau,
        direction=direction,
    )


def level_crossing_time(
    time_seconds: np.ndarray,
    signal: np.ndarray,
    level: float,
    *,
    after_time_seconds: float | None = None,
) -> float | None:
    time_array = np.asarray(time_seconds, dtype=float)
    signal_array = np.asarray(signal, dtype=float)
    if time_array.ndim != 1 or signal_array.ndim != 1 or time_array.shape != signal_array.shape:
        raise ValueError("time_seconds and signal must be 1D arrays with the same shape.")

    if after_time_seconds is not None:
        start = int(np.searchsorted(time_array, after_time_seconds, side="left"))
        time_array = time_array[start:]
        signal_array = signal_array[start:]
        if time_array.size < 2:
            return None

    crossings = np.flatnonzero((signal_array[:-1] <= level) & (signal_array[1:] >= level))
    if crossings.size == 0:
        return None

    index = int(crossings[0])
    x0 = time_array[index]
    x1 = time_array[index + 1]
    y0 = signal_array[index]
    y1 = signal_array[index + 1]
    if np.isclose(y1, y0):
        return float(x0)
    weight = (level - y0) / (y1 - y0)
    return float(x0 + weight * (x1 - x0))


def measure_rise_time(
    time_seconds: np.ndarray,
    signal: np.ndarray,
    *,
    low: float = 0.1,
    high: float = 0.9,
    after_time_seconds: float | None = None,
) -> dict[str, float | None]:
    t_low = level_crossing_time(
        time_seconds,
        signal,
        low,
        after_time_seconds=after_time_seconds,
    )
    t_high = level_crossing_time(
        time_seconds,
        signal,
        high,
        after_time_seconds=after_time_seconds,
    )
    rise = None if t_low is None or t_high is None else float(t_high - t_low)
    return {
        "low": float(low),
        "high": float(high),
        "t_low_seconds": t_low,
        "t_high_seconds": t_high,
        "rise_time_seconds": rise,
    }
