from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..core.arrays import ArrayLike1D, _as_1d_array, _interp_complex


def prefix_sum(values: ArrayLike1D) -> np.ndarray:
    return np.cumsum(np.asarray(values))


def estimate_delay_from_phase(freqs: ArrayLike1D, s21: ArrayLike1D) -> float:
    freqs_array = _as_1d_array(freqs, name="freqs", dtype=float)
    s21_array = _as_1d_array(s21, name="s21", dtype=complex)
    if freqs_array.shape != s21_array.shape:
        raise ValueError("freqs and s21 must have the same shape.")
    slope, _ = np.polyfit(freqs_array, np.unwrap(np.angle(s21_array)), 1)
    return -slope / (2 * np.pi)


def group_delay_from_s21(
    s21: ArrayLike1D,
    *,
    freqs: ArrayLike1D | None = None,
    dfreq: float | None = None,
) -> np.ndarray:
    s21_array = _as_1d_array(s21, name="s21", dtype=complex)
    if freqs is None:
        if dfreq is None:
            raise ValueError("Provide either freqs or dfreq.")
        freq_axis = np.arange(s21_array.size, dtype=float) * float(dfreq)
    else:
        freq_axis = _as_1d_array(freqs, name="freqs", dtype=float)
        if freq_axis.shape != s21_array.shape:
            raise ValueError("freqs and s21 must have the same shape.")

    spline = CubicSpline(freq_axis, np.unwrap(np.angle(s21_array)), extrapolate=True)
    return -spline(freq_axis, 1) / (2 * np.pi)


def impulse_to_step_response(
    impulse: ArrayLike1D,
    dt: float,
    *,
    initial_value: float = 0.0,
) -> np.ndarray:
    impulse_array = np.asarray(impulse)
    return initial_value + np.cumsum(impulse_array) * float(dt)


def s21_to_impulse_response(
    s21: ArrayLike1D,
    dfreq: float,
    *,
    extend_multiple: int = 5,
    interp_multiple: int = 10,
    inter_multiple: int | None = None,
    linear_extend: bool = False,
    is_linear_extend: bool | None = None,
    return_full_spectrum: bool = False,
    is_return_S21_full: bool | None = None,
    linear_extend_ref: int = 200,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    if inter_multiple is not None:
        interp_multiple = inter_multiple
    if is_linear_extend is not None:
        linear_extend = is_linear_extend
    if is_return_S21_full is not None:
        return_full_spectrum = is_return_S21_full

    if extend_multiple < 1:
        raise ValueError("extend_multiple must be >= 1.")
    if interp_multiple < 1:
        raise ValueError("interp_multiple must be >= 1.")
    if dfreq <= 0:
        raise ValueError("dfreq must be positive.")

    s21_array = _as_1d_array(s21, name="s21", dtype=complex)
    original_index = np.arange(s21_array.size, dtype=float)
    interp_index = np.linspace(0.0, s21_array.size - 1, s21_array.size * interp_multiple)
    s21_interp = _interp_complex(interp_index, original_index, s21_array)

    if linear_extend:
        ref = min(int(linear_extend_ref), s21_interp.size)
        x_tail = np.arange(s21_interp.size - ref, s21_interp.size, dtype=float)
        log_mag = np.log(np.maximum(np.abs(s21_interp[-ref:]), 1e-15))
        unwrapped_phase = np.unwrap(np.angle(s21_interp[-ref:]))
        mag_coeffs = np.polyfit(x_tail, log_mag, 1)
        phase_coeffs = np.polyfit(x_tail, unwrapped_phase, 1)

        target_size = s21_interp.size * extend_multiple
        x_full = np.arange(target_size, dtype=float)
        log_mag_full = np.polyval(mag_coeffs, x_full)
        phase_full = np.polyval(phase_coeffs, x_full)
        extrapolated = np.exp(log_mag_full + 1j * phase_full)
        extrapolated[: s21_interp.size] = s21_interp
        s21_positive = extrapolated
        full_length = 2 * s21_positive.size
    else:
        s21_positive = s21_interp
        full_length = 2 * extend_multiple * s21_positive.size

    spectrum = np.zeros(full_length, dtype=complex)
    spectrum[: s21_positive.size] = s21_positive
    spectrum[-(s21_positive.size - 1) :] = np.conj(s21_positive[1:][::-1])

    df = float(dfreq) / interp_multiple
    impulse = np.fft.fftshift(np.fft.ifft(spectrum) * spectrum.size * df)
    t = np.linspace(-0.5 / df, 0.5 / df, spectrum.size, endpoint=False)

    if return_full_spectrum:
        return t, impulse, spectrum
    return t, impulse
