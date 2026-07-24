from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import curve_fit

from ..core.arrays import ArrayLike1D, _as_1d_array
from .multiexp import simple_multiexp_s21, simple_multiexp_step_response
from .transforms import estimate_delay_from_phase


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
