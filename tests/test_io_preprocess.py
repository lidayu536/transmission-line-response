from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from transmission_line_response import PreprocessConfig, RawS21Data, prepare_s21, read_touchstone_2port


class IoPreprocessTests(unittest.TestCase):
    def test_read_touchstone_db_phase_s21(self) -> None:
        content = """# GHz S DB R 50
1.0 -20 0 -10 -90 -30 45 -20 180
2.0 -21 0 -12 -180 -31 45 -21 180
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.s2p"
            path.write_text(content, encoding="utf-8")
            data = read_touchstone_2port(path)

        self.assertEqual(data.data_format, "DB")
        np.testing.assert_allclose(data.freq_hz, [1.0e9, 2.0e9])
        np.testing.assert_allclose(np.abs(data.s21), 10 ** (np.array([-10.0, -12.0]) / 20.0))
        np.testing.assert_allclose(np.rad2deg(np.angle(data.s21)), [-90.0, -180.0], atol=1e-12)

    def test_prepare_s21_adds_dc_and_halves_roundtrip(self) -> None:
        freq_hz = np.array([1.0e9, 2.0e9, 3.0e9])
        mag_db = np.array([-20.0, -22.0, -24.0])
        phase = np.array([-0.4, -0.8, -1.2])
        raw = RawS21Data(freq_hz=freq_hz, s21=10 ** (mag_db / 20.0) * np.exp(1j * phase))
        config = PreprocessConfig(
            grid_stop_hz=3.0e9,
            grid_step_hz=1.0e9,
            roundtrip_to_single_pass=True,
        )
        prepared = prepare_s21(raw, config)

        np.testing.assert_allclose(prepared.freq_hz, [0.0, 1.0e9, 2.0e9])
        self.assertTrue(prepared.dc_added)
        self.assertAlmostEqual(prepared.dc_magnitude_db, -18.0)
        np.testing.assert_allclose(prepared.s21_db, [-9.0, -10.0, -11.0])
        np.testing.assert_allclose(prepared.phase_rad, [0.0, -0.2, -0.4], atol=1e-12)


if __name__ == "__main__":
    unittest.main()