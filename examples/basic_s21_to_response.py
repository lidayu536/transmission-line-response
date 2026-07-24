"""Minimal synthetic S21-to-response example.

Run from the repository root with:

    python examples/basic_s21_to_response.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from transmission_line_response import IfftConfig, PreprocessConfig, RawS21Data, analyze_rise_from_s21


def fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


freq_hz = np.arange(0.0, 5.0e9, 5.0e6)
freq_ghz = freq_hz / 1.0e9

# A simple one-pole low-pass line with 3 ns propagation delay.
tau_ns = 0.35
delay_ns = 3.0
s21 = np.exp(-1j * 2.0 * np.pi * freq_ghz * delay_ns) / (1.0 + 1j * 2.0 * np.pi * freq_ghz * tau_ns)

raw = RawS21Data(freq_hz=freq_hz, s21=s21, source_format="synthetic")
response = analyze_rise_from_s21(
    raw,
    preprocess_config=PreprocessConfig(add_dc_if_needed=False, roundtrip_to_single_pass=False),
    ifft_config=IfftConfig(align_offset_ns=0.0, normalization_mode="reference_point", normalization_time_ns=20.0),
    label="synthetic one-pole line",
)

print(f"t10_ns={fmt(response.metrics.t10_ns)}")
print(f"t90_ns={fmt(response.metrics.t90_ns)}")
print(f"rise_10_90_ns={fmt(response.metrics.rise_10_90_ns)}")