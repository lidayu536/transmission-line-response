from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawS21Data:
    """Raw S21 samples after reading a measurement file."""

    freq_hz: np.ndarray
    s21: np.ndarray
    path: Path | None = None
    source_format: str = "array"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.path.stem if self.path is not None else "S21"


@dataclass(frozen=True)
class PreparedS21:
    """S21 placed on a zero-start uniform grid, ready for phase/time analysis."""

    label: str
    raw: RawS21Data
    freq_hz: np.ndarray
    freq_ghz: np.ndarray
    fullpass_db: np.ndarray
    fullpass_phase_rad: np.ndarray
    s21_db: np.ndarray
    phase_rad: np.ndarray
    s21: np.ndarray
    single_pass_assumed: bool
    dc_added: bool
    dc_magnitude_db: float | None
    grid_step_hz: float
    notes: tuple[str, ...] = ()

    @property
    def source_path(self) -> Path | None:
        return self.raw.path

    @property
    def source_format(self) -> str:
        return self.raw.source_format
