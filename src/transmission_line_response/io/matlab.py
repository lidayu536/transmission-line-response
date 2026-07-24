from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..core import RawS21Data
from .common import _as_1d, _sanitize_s21


def _pick_key(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {key.lower(): key for key in keys}
    for candidate in candidates:
        found = lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def read_mat_s21(path: Path):
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    visible_keys = [key for key in mat if not key.startswith("__")]
    freq_key = _pick_key(visible_keys, ["Freq", "freq", "frequency", "Frequency", "f"])
    s21_key = _pick_key(visible_keys, ["S21", "s21", "S_21", "s21_complex"])

    if freq_key is None:
        raise ValueError(f"Cannot find a frequency vector in {path}. Available keys: {visible_keys}")
    freq_hz = _as_1d(mat[freq_key], name=freq_key, dtype=float)

    if s21_key is not None:
        raw_s21 = np.asarray(mat[s21_key]).reshape(-1)
        if np.iscomplexobj(raw_s21):
            s21 = raw_s21.astype(complex)
        else:
            mag_db_key = _pick_key(visible_keys, ["S21dB", "s21db", "S21_db", "s21_db"])
            phase_key = _pick_key(visible_keys, ["S21phase", "s21phase", "S21_phase", "phase"])
            if mag_db_key is None or phase_key is None:
                raise ValueError(f"{path} has a real S21 array; provide S21dB and S21phase keys.")
            mag_db = _as_1d(mat[mag_db_key], name=mag_db_key, dtype=float)
            phase = _as_1d(mat[phase_key], name=phase_key, dtype=float)
            phase_rad = np.deg2rad(phase) if np.nanmax(np.abs(phase)) > 2.0 * np.pi else phase
            s21 = np.power(10.0, mag_db / 20.0) * np.exp(1j * phase_rad)
    else:
        mag_db_key = _pick_key(visible_keys, ["S21dB", "s21db", "S21_db", "s21_db"])
        phase_key = _pick_key(visible_keys, ["S21phase", "s21phase", "S21_phase", "phase"])
        if mag_db_key is None or phase_key is None:
            raise ValueError(f"Cannot find S21 in {path}. Available keys: {visible_keys}")
        mag_db = _as_1d(mat[mag_db_key], name=mag_db_key, dtype=float)
        phase = _as_1d(mat[phase_key], name=phase_key, dtype=float)
        phase_rad = np.deg2rad(phase) if np.nanmax(np.abs(phase)) > 2.0 * np.pi else phase
        s21 = np.power(10.0, mag_db / 20.0) * np.exp(1j * phase_rad)

    freq_hz, s21 = _sanitize_s21(freq_hz, _as_1d(s21, name="S21", dtype=complex))
    return RawS21Data(
        freq_hz=freq_hz,
        s21=s21,
        path=path,
        source_format=".mat",
        metadata={"mat_keys": tuple(visible_keys), "freq_key": freq_key, "s21_key": s21_key},
    )
