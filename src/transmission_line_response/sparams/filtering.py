from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from ..core.arrays import ArrayLike1D, _as_1d_array


def filter_waveform_by_s21(
    waveform: ArrayLike1D,
    dt: float,
    s21: ArrayLike1D,
    dfreq: float,
    *,
    idle_extend: float = 0.0,
    trim: bool = True,
) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if dfreq <= 0:
        raise ValueError("dfreq must be positive.")

    waveform_array = _as_1d_array(waveform, name="waveform", dtype=complex)
    s21_array = _as_1d_array(s21, name="s21", dtype=complex)

    pad = 0
    if idle_extend > 0.0:
        pad = int(np.ceil(idle_extend / dt))
        waveform_work = np.pad(waveform_array, (pad, pad), mode="constant")
    else:
        waveform_work = waveform_array

    fft_freqs = np.fft.fftfreq(waveform_work.size, d=dt)
    positive_freqs = np.arange(s21_array.size, dtype=float) * dfreq
    transfer_interp = interp1d(
        positive_freqs,
        s21_array,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    transfer = transfer_interp(np.abs(fft_freqs))
    transfer[fft_freqs < 0] = np.conj(transfer[fft_freqs < 0])

    filtered = np.fft.ifft(np.fft.fft(waveform_work) * transfer)
    if np.isrealobj(waveform):
        filtered = filtered.real
    if trim and pad > 0:
        return filtered[pad : pad + waveform_array.size]
    return filtered


filter_by_S21 = filter_waveform_by_s21
