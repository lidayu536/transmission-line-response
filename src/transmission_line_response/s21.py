from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit


ArrayLike1D = Sequence[float] | Sequence[complex] | np.ndarray


def _as_1d_array(values: ArrayLike1D, *, name: str, dtype: np.dtype | type) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _interp_complex(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, np.real(y)) + 1j * np.interp(x_new, x, np.imag(y))


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


def simple_multiexp_impulse_response(
    t: ArrayLike1D,
    alphas: ArrayLike1D,
    taus: ArrayLike1D,
) -> np.ndarray:
    t_array = np.asarray(t, dtype=float)
    alphas_array = _as_1d_array(alphas, name="alphas", dtype=float)
    taus_array = _as_1d_array(taus, name="taus", dtype=float)
    if alphas_array.shape != taus_array.shape:
        raise ValueError("alphas and taus must have the same shape.")
    if np.any(taus_array <= 0):
        raise ValueError("taus must be strictly positive.")
    return np.sum(
        alphas_array[:, None] * np.exp(-t_array[None, :] / taus_array[:, None]),
        axis=0,
    )


def simple_multiexp_step_response(
    t: ArrayLike1D,
    alphas: ArrayLike1D,
    taus: ArrayLike1D,
    *,
    normalize: bool = True,
) -> np.ndarray:
    t_array = np.asarray(t, dtype=float)
    alphas_array = _as_1d_array(alphas, name="alphas", dtype=float)
    taus_array = _as_1d_array(taus, name="taus", dtype=float)
    if alphas_array.shape != taus_array.shape:
        raise ValueError("alphas and taus must have the same shape.")
    if np.any(taus_array <= 0):
        raise ValueError("taus must be strictly positive.")

    weights = alphas_array.astype(float)
    if normalize:
        total = np.sum(weights)
        if np.isclose(total, 0.0):
            raise ValueError("Cannot normalize alphas because their sum is zero.")
        weights = weights / total

    return np.sum(
        weights[:, None] * (1.0 - np.exp(-t_array[None, :] / taus_array[:, None])),
        axis=0,
    )


def simple_multiexp_s21(
    freqs: ArrayLike1D,
    alphas: ArrayLike1D,
    taus: ArrayLike1D,
    delay: float = 0.0,
) -> np.ndarray:
    freqs_array = np.asarray(freqs, dtype=float)
    alphas_array = _as_1d_array(alphas, name="alphas", dtype=float)
    taus_array = _as_1d_array(taus, name="taus", dtype=float)
    if alphas_array.shape != taus_array.shape:
        raise ValueError("alphas and taus must have the same shape.")
    if np.any(taus_array <= 0):
        raise ValueError("taus must be strictly positive.")

    terms = np.sum(
        alphas_array[:, None] / (1.0 + 1j * 2 * np.pi * freqs_array[None, :] * taus_array[:, None]),
        axis=0,
    )
    return np.exp(-1j * 2 * np.pi * freqs_array * delay) * terms


def oscillatory_multiexp_impulse_response(
    t: ArrayLike1D,
    alphas: ArrayLike1D,
    taus: ArrayLike1D,
    omegas: ArrayLike1D,
    phis: ArrayLike1D,
) -> np.ndarray:
    t_array = np.asarray(t, dtype=float)
    alphas_array = _as_1d_array(alphas, name="alphas", dtype=float)
    taus_array = _as_1d_array(taus, name="taus", dtype=float)
    omegas_array = _as_1d_array(omegas, name="omegas", dtype=float)
    phis_array = _as_1d_array(phis, name="phis", dtype=float)
    if not (
        alphas_array.shape == taus_array.shape == omegas_array.shape == phis_array.shape
    ):
        raise ValueError("alphas, taus, omegas, and phis must have the same shape.")
    if np.any(taus_array <= 0):
        raise ValueError("taus must be strictly positive.")

    return np.sum(
        alphas_array[:, None]
        * np.exp(-t_array[None, :] / taus_array[:, None])
        * np.cos(omegas_array[:, None] * t_array[None, :] + phis_array[:, None]),
        axis=0,
    )


def oscillatory_multiexp_s21(
    freqs: ArrayLike1D,
    alphas: ArrayLike1D,
    taus: ArrayLike1D,
    omegas: ArrayLike1D,
    phis: ArrayLike1D,
    delay: float = 0.0,
) -> np.ndarray:
    freqs_array = np.asarray(freqs, dtype=float)
    alphas_array = _as_1d_array(alphas, name="alphas", dtype=float)
    taus_array = _as_1d_array(taus, name="taus", dtype=float)
    omegas_array = _as_1d_array(omegas, name="omegas", dtype=float)
    phis_array = _as_1d_array(phis, name="phis", dtype=float)
    if not (
        alphas_array.shape == taus_array.shape == omegas_array.shape == phis_array.shape
    ):
        raise ValueError("alphas, taus, omegas, and phis must have the same shape.")
    if np.any(taus_array <= 0):
        raise ValueError("taus must be strictly positive.")

    s = 1j * 2 * np.pi * freqs_array
    numerator = np.sum(
        alphas_array[:, None]
        * (
            np.cos(phis_array[:, None]) * (s[None, :] + 1.0 / taus_array[:, None])
            / ((s[None, :] + 1.0 / taus_array[:, None]) ** 2 + omegas_array[:, None] ** 2)
            - np.sin(phis_array[:, None]) * omegas_array[:, None]
            / ((s[None, :] + 1.0 / taus_array[:, None]) ** 2 + omegas_array[:, None] ** 2)
        ),
        axis=0,
    )
    return np.exp(-1j * 2 * np.pi * freqs_array * delay) * (1.0 - s * numerator)


def standing_wave_s21(
    freqs: ArrayLike1D,
    alphas: ArrayLike1D,
    taus: ArrayLike1D,
    *,
    delay: float = 0.0,
    roundtrip: float,
    rho_mag: float,
    phi: float = 0.0,
) -> np.ndarray:
    if abs(rho_mag) >= 1:
        raise ValueError("rho_mag must satisfy |rho_mag| < 1 for a convergent echo series.")
    base = simple_multiexp_s21(freqs, alphas, taus, delay=delay)
    freqs_array = np.asarray(freqs, dtype=float)
    denominator = 1.0 - rho_mag * np.exp(1j * phi) * np.exp(-1j * 2 * np.pi * freqs_array * roundtrip)
    return base / denominator


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


@dataclass(frozen=True)
class SimpleMultiexpFitResult:
    alphas: np.ndarray
    taus: np.ndarray
    delay: float
    sigma_alphas: np.ndarray
    sigma_taus: np.ndarray
    sigma_delay: float
    covariance: np.ndarray

    @property
    def alpha_sum(self) -> float:
        return float(np.sum(self.alphas))

    def normalized_alphas(self) -> np.ndarray:
        if np.isclose(self.alpha_sum, 0.0):
            raise ValueError("Cannot normalize alphas because their sum is zero.")
        return self.alphas / self.alpha_sum

    def s21(self, freqs: ArrayLike1D) -> np.ndarray:
        return simple_multiexp_s21(freqs, self.alphas, self.taus, self.delay)

    def step_response(self, t: ArrayLike1D, *, normalize: bool = True) -> np.ndarray:
        return simple_multiexp_step_response(t, self.alphas, self.taus, normalize=normalize)


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


def _default_initial_taus(freqs: np.ndarray, order: int) -> np.ndarray:
    positive = np.abs(freqs[np.abs(freqs) > 0])
    if positive.size == 0:
        base_tau = 1.0
    else:
        base_tau = 1.0 / (2 * np.pi * np.max(positive))
    if order == 1:
        return np.array([base_tau * 10.0], dtype=float)
    return np.geomspace(base_tau, base_tau * 100.0, order)


def fit_simple_multiexp(
    freqs: ArrayLike1D,
    s21: ArrayLike1D,
    *,
    order: int = 1,
    alpha_bound: tuple[float, float] = (-1.0, 1.0),
    tau_bound: tuple[float, float] = (0.0, np.inf),
    delay_bound: tuple[float, float] = (0.0, np.inf),
    initial_alphas: ArrayLike1D | None = None,
    initial_taus: ArrayLike1D | None = None,
    initial_delay: float | None = None,
    maxfev: int = 50000,
) -> SimpleMultiexpFitResult:
    if order < 1:
        raise ValueError("order must be >= 1.")

    freqs_array = _as_1d_array(freqs, name="freqs", dtype=float)
    s21_array = _as_1d_array(s21, name="s21", dtype=complex)
    if freqs_array.shape != s21_array.shape:
        raise ValueError("freqs and s21 must have the same shape.")

    if initial_alphas is None:
        dc_gain = float(np.abs(s21_array[0])) if s21_array.size else 1.0
        initial_alphas_array = np.full(order, dc_gain / order, dtype=float)
    else:
        initial_alphas_array = _as_1d_array(initial_alphas, name="initial_alphas", dtype=float)

    if initial_taus is None:
        initial_taus_array = _default_initial_taus(freqs_array, order)
    else:
        initial_taus_array = _as_1d_array(initial_taus, name="initial_taus", dtype=float)

    if initial_alphas_array.size != order or initial_taus_array.size != order:
        raise ValueError("initial_alphas and initial_taus must match order.")

    if initial_delay is None:
        initial_delay = estimate_delay_from_phase(freqs_array, s21_array)

    def fit_func(fs: np.ndarray, *params: float) -> np.ndarray:
        alphas_fit = np.asarray(params[:order], dtype=float)
        taus_fit = np.asarray(params[order : 2 * order], dtype=float)
        delay_fit = float(params[-1])
        modeled = simple_multiexp_s21(fs, alphas_fit, taus_fit, delay_fit)
        return np.concatenate([np.real(modeled), np.imag(modeled)])

    p0 = np.concatenate([initial_alphas_array, initial_taus_array, [initial_delay]])
    lower = np.array([alpha_bound[0]] * order + [tau_bound[0]] * order + [delay_bound[0]], dtype=float)
    upper = np.array([alpha_bound[1]] * order + [tau_bound[1]] * order + [delay_bound[1]], dtype=float)

    params, covariance = curve_fit(
        fit_func,
        freqs_array,
        np.concatenate([np.real(s21_array), np.imag(s21_array)]),
        p0=p0,
        bounds=(lower, upper),
        maxfev=maxfev,
    )

    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    return SimpleMultiexpFitResult(
        alphas=np.asarray(params[:order], dtype=float),
        taus=np.asarray(params[order : 2 * order], dtype=float),
        delay=float(params[-1]),
        sigma_alphas=np.asarray(sigma[:order], dtype=float),
        sigma_taus=np.asarray(sigma[order : 2 * order], dtype=float),
        sigma_delay=float(sigma[-1]),
        covariance=covariance,
    )


def find_step_response_care_points(
    times: ArrayLike1D,
    response: ArrayLike1D,
    *,
    care_ratios: Iterable[float] = (),
    care_times: Iterable[float] = (),
    response_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[tuple[float, float]]:
    times_array = _as_1d_array(times, name="times", dtype=float)
    response_array = _as_1d_array(response, name="response", dtype=float)
    if times_array.shape != response_array.shape:
        raise ValueError("times and response must have the same shape.")

    points: list[tuple[float, float]] = []
    for ratio in care_ratios:
        index = int(np.argmin(np.abs(response_array - ratio)))
        points.append((float(times_array[index]), float(response_array[index])))

    for care_time in care_times:
        if response_fn is None:
            value = np.interp(care_time, times_array, response_array)
        else:
            value = float(np.asarray(response_fn(np.asarray([care_time], dtype=float)))[0])
        points.append((float(care_time), float(value)))
    return points


def fit_s21_with_simple_multiexp(
    freqs: ArrayLike1D,
    s21: ArrayLike1D,
    order: int = 1,
    alpha_bound: tuple[float, float] = (-1.0, 1.0),
    tau_bound: tuple[float, float] = (0.0, np.inf),
):
    fit = fit_simple_multiexp(
        freqs,
        s21,
        order=order,
        alpha_bound=alpha_bound,
        tau_bound=tau_bound,
    )
    return (
        fit.alphas,
        fit.taus,
        fit.delay,
    ), (
        fit.sigma_alphas,
        fit.sigma_taus,
        fit.sigma_delay,
    )


def calibrate_s21_by_simple_multiexp(
    freqs: ArrayLike1D,
    s21: ArrayLike1D,
    order: int = 1,
    alpha_bound: tuple[float, float] = (-1.0, 1.0),
    tau_bound: tuple[float, float] = (0.0, np.inf),
    freq_unit: str = "GHz",
    time_unit: str = "ns",
    care_ratios: Iterable[float] = (0.95,),
    care_times: Iterable[float] = (),
    verbose: bool = False,
):
    fit = fit_simple_multiexp(
        freqs,
        s21,
        order=order,
        alpha_bound=alpha_bound,
        tau_bound=tau_bound,
    )

    step_fn = lambda t: fit.step_response(np.asarray(t, dtype=float), normalize=True)
    max_tau = float(np.max(fit.taus))
    max_care_time = max([0.0, *[float(x) for x in care_times]])
    t_max = max(max_tau * 6.0, max_care_time * 1.05, max_tau)
    time_grid = np.linspace(0.0, t_max, 1001)
    step_grid = step_fn(time_grid)
    care_points = find_step_response_care_points(
        time_grid,
        step_grid,
        care_ratios=care_ratios,
        care_times=care_times,
        response_fn=step_fn,
    )

    if verbose:
        print(f"delay ({time_unit}) = {fit.delay:.6g} +/- {fit.sigma_delay:.3g}")
        print(f"frequency unit = {freq_unit}")
        for alpha, sigma_alpha, tau, sigma_tau in zip(
            fit.alphas,
            fit.sigma_alphas,
            fit.taus,
            fit.sigma_taus,
        ):
            print(
                f"alpha={alpha:.6g} +/- {sigma_alpha:.3g}, "
                f"tau ({time_unit})={tau:.6g} +/- {sigma_tau:.3g}"
            )

    return step_fn, {
        "alphas": fit.alphas,
        "taus": fit.taus,
        "delay": fit.delay,
        "sigma_alphas": fit.sigma_alphas,
        "sigma_taus": fit.sigma_taus,
        "sigma_delay": fit.sigma_delay,
        "care_points": care_points,
        "time_unit": time_unit,
        "freq_unit": freq_unit,
    }


# Notebook-compatible aliases.
reponse_simple_multiexp = simple_multiexp_impulse_response
s21_simple_multiexp = simple_multiexp_s21
filter_by_S21 = filter_waveform_by_s21
