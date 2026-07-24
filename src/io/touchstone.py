from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FREQ_SCALES = {
    "HZ": 1.0,
    "KHZ": 1e3,
    "MHZ": 1e6,
    "GHZ": 1e9,
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
