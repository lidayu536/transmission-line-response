from __future__ import annotations

import numpy as np

from ..core.arrays import ArrayLike1D, _as_1d_array


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


reponse_simple_multiexp = simple_multiexp_impulse_response
s21_simple_multiexp = simple_multiexp_s21
