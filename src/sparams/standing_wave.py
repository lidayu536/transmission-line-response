from __future__ import annotations

import numpy as np

from ..core.arrays import ArrayLike1D
from .multiexp import simple_multiexp_s21


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
