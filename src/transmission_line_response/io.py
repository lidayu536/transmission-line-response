from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_FREQ_SCALES = {
    "HZ": 1.0,
    "KHZ": 1e3,
    "MHZ": 1e6,
    "GHZ": 1e9,
}

_TIME_SCALES = {
    "S": 1.0,
    "MS": 1e-3,
    "US": 1e-6,
    "NS": 1e-9,
    "PS": 1e-12,
    "FS": 1e-15,
}


@dataclass(frozen=True)
class Touchstone2Port:
    path: Path
    freq_hz: np.ndarray
    s11: np.ndarray
    s21: np.ndarray
    s12: np.ndarray
    s22: np.ndarray
    data_format: str
    reference_ohms: float

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {
            "S11": self.s11,
            "S21": self.s21,
            "S12": self.s12,
            "S22": self.s22,
        }


@dataclass(frozen=True)
class ScopeTrace:
    path: Path
    time_seconds: np.ndarray
    signal: np.ndarray
    time_column: str
    value_column: str


def _decode_touchstone_pair(values: np.ndarray, data_format: str) -> np.ndarray:
    fmt = data_format.upper()
    first = values[:, 0]
    second = values[:, 1]
    if fmt == "RI":
        return first + 1j * second
    if fmt == "MA":
        return first * np.exp(1j * np.deg2rad(second))
    if fmt == "DB":
        return np.power(10.0, first / 20.0) * np.exp(1j * np.deg2rad(second))
    raise ValueError(f"Unsupported Touchstone data format: {data_format}")


def read_touchstone_2port(path: str | Path) -> Touchstone2Port:
    path_obj = Path(path)
    header_tokens: list[str] | None = None
    numeric_tokens: list[float] = []

    for raw_line in path_obj.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            header_tokens = line[1:].split()
            continue
        if "!" in line:
            line = line.split("!", 1)[0].strip()
        if not line:
            continue
        numeric_tokens.extend(float(token) for token in line.split())

    if header_tokens is None:
        raise ValueError(f"Touchstone header line not found in {path_obj}")
    if len(header_tokens) < 3:
        raise ValueError(f"Invalid Touchstone header in {path_obj}: {header_tokens}")

    freq_unit = header_tokens[0].upper()
    network_type = header_tokens[1].upper()
    data_format = header_tokens[2].upper()
    reference_ohms = 50.0
    if len(header_tokens) >= 5 and header_tokens[3].upper() == "R":
        reference_ohms = float(header_tokens[4])

    if network_type != "S":
        raise ValueError(f"Only S-parameter Touchstone files are supported: {path_obj}")
    if freq_unit not in _FREQ_SCALES:
        raise ValueError(f"Unsupported Touchstone frequency unit: {freq_unit}")
    if len(numeric_tokens) % 9 != 0:
        raise ValueError(f"Expected Touchstone data in 9-value blocks, got {len(numeric_tokens)} values.")

    data = np.asarray(numeric_tokens, dtype=float).reshape(-1, 9)
    freq_hz = data[:, 0] * _FREQ_SCALES[freq_unit]
    return Touchstone2Port(
        path=path_obj,
        freq_hz=freq_hz,
        s11=_decode_touchstone_pair(data[:, 1:3], data_format),
        s21=_decode_touchstone_pair(data[:, 3:5], data_format),
        s12=_decode_touchstone_pair(data[:, 5:7], data_format),
        s22=_decode_touchstone_pair(data[:, 7:9], data_format),
        data_format=data_format,
        reference_ohms=reference_ohms,
    )


def read_s2p(path: str | Path) -> Touchstone2Port:
    return read_touchstone_2port(path)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def _guess_time_column(fieldnames: list[str]) -> str:
    normalized = {_normalize_column_name(name): name for name in fieldnames}
    for key in ("t", "time", "time(s)", "time(ns)", "time(us)", "time(ms)", "x"):
        if key in normalized:
            return normalized[key]
    for name in fieldnames:
        if "time" in _normalize_column_name(name):
            return name
    return fieldnames[0]


def _guess_value_column(fieldnames: list[str], time_column: str) -> str:
    normalized = {_normalize_column_name(name): name for name in fieldnames}
    for key in ("z", "v", "voltage", "signal", "y"):
        if key in normalized:
            return normalized[key]
    for name in fieldnames:
        if name != time_column:
            return name
    raise ValueError("Could not infer a value column from scope CSV.")


def _time_scale_from_column(name: str) -> float:
    match = re.search(r"\(([^)]+)\)", name)
    if not match:
        return 1.0
    unit = match.group(1).strip().upper()
    return _TIME_SCALES.get(unit, 1.0)


def read_scope_csv(
    path: str | Path,
    *,
    time_column: str | None = None,
    value_column: str | None = None,
) -> ScopeTrace:
    path_obj = Path(path)
    with path_obj.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path_obj}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        time_name = time_column or _guess_time_column(fieldnames)
        value_name = value_column or _guess_value_column(fieldnames, time_name)
        time_values: list[float] = []
        signal_values: list[float] = []
        for row in reader:
            if row[time_name] == "" or row[value_name] == "":
                continue
            time_values.append(float(row[time_name]))
            signal_values.append(float(row[value_name]))

    time_scale = _time_scale_from_column(time_name)
    return ScopeTrace(
        path=path_obj,
        time_seconds=np.asarray(time_values, dtype=float) * time_scale,
        signal=np.asarray(signal_values, dtype=float),
        time_column=time_name,
        value_column=value_name,
    )


def subtract_scope_background(trace: ScopeTrace, background: ScopeTrace) -> ScopeTrace:
    background_interp = np.interp(
        trace.time_seconds,
        background.time_seconds,
        background.signal,
    )
    return ScopeTrace(
        path=trace.path,
        time_seconds=trace.time_seconds.copy(),
        signal=trace.signal - background_interp,
        time_column=trace.time_column,
        value_column=trace.value_column,
    )


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


def _pick_key(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {key.lower(): key for key in keys}
    for candidate in candidates:
        found = lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def _read_mat_s21(path: Path):
    from scipy.io import loadmat

    from .types import RawS21Data

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


def read_s21(path: str | Path):
    """Read S21 from a supported local measurement file.

    Supported formats are currently Touchstone `.s2p` and MATLAB `.mat`.
    """

    from .types import RawS21Data

    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".s2p":
        network = read_s2p(path_obj)
        freq_hz, s21 = _sanitize_s21(network.freq_hz.astype(float), network.s21.astype(complex))
        return RawS21Data(
            freq_hz=freq_hz,
            s21=s21,
            path=path_obj,
            source_format=".s2p",
            metadata={
                "touchstone_format": network.data_format,
                "reference_ohms": network.reference_ohms,
            },
        )
    if suffix == ".mat":
        return _read_mat_s21(path_obj)
    raise ValueError(f"Unsupported S21 file type: {path_obj.suffix}")
