from __future__ import annotations

import unittest
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

import numpy as np

from transmission_line_response import group_delay_from_s21, measure_rise_time


class TimePhaseTests(unittest.TestCase):
    def test_group_delay_tracks_linear_phase_delay(self) -> None:
        freq_ghz = np.linspace(0.0, 1.0, 501)
        delay_ns = 3.25
        s21 = np.exp(-1j * 2.0 * np.pi * freq_ghz * delay_ns)
        gd_ns = group_delay_from_s21(s21, freqs=freq_ghz)
        np.testing.assert_allclose(gd_ns[20:-20], delay_ns, atol=1e-10)

    def test_measure_rise_time_on_linear_edge(self) -> None:
        time = np.linspace(0.0, 8.0, 801)
        signal = np.clip((time - 1.0) / 4.0, 0.0, 1.0)
        result = measure_rise_time(time, signal)
        self.assertAlmostEqual(result["t_low_seconds"], 1.4, places=6)
        self.assertAlmostEqual(result["t_high_seconds"], 4.6, places=6)
        self.assertAlmostEqual(result["rise_time_seconds"], 3.2, places=6)


if __name__ == "__main__":
    unittest.main()