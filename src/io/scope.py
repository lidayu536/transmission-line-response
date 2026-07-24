from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TIME_SCALES = {
    "S": 1.0,
    "MS": 1e-3,
    "US": 1e-6,
    "NS": 1e-9,
    "PS": 1e-12,
    "FS": 1e-15,
}


@dataclass(frozen=True)
class ScopeTrace:
    path: Path
    time_seconds: np.ndarray
    signal: np.ndarray
    time_column: str
    value_column: str


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
