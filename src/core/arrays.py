from __future__ import annotations

from typing import Sequence

import numpy as np

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
