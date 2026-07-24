from __future__ import annotations

from pathlib import Path

from ..core import RawS21Data
from .common import _sanitize_s21
from .matlab import read_mat_s21
from .touchstone import read_s2p


def read_s21(path: str | Path):
    """Read S21 from a supported local measurement file.

    Supported formats are currently Touchstone `.s2p` and MATLAB `.mat`.
    """

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
        return read_mat_s21(path_obj)
    raise ValueError(f"Unsupported S21 file type: {path_obj.suffix}")
