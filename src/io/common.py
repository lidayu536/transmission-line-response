from __future__ import annotations

import numpy as np


def _as_1d(values: object, *, name: str, dtype: type) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty.")
    return array


def _sanitize_s21(freq_hz: np.ndarray, s21: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if freq_hz.shape != s21.shape:
        raise ValueError("freq_hz and s21 must have the same shape.")
    mask = np.isfinite(freq_hz) & np.isfinite(np.real(s21)) & np.isfinite(np.imag(s21))
    freq_work = freq_hz[mask].astype(float)
    s21_work = s21[mask].astype(complex)
    order = np.argsort(freq_work)
    freq_work = freq_work[order]
    s21_work = s21_work[order]
    unique_freq, unique_idx = np.unique(freq_work, return_index=True)
    return unique_freq, s21_work[unique_idx]
