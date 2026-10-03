from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from transmission_line_response import (  # noqa: E402
    design_interpolated_ripple_correction_fir,
    design_fir_sequence_from_s21,
    frequency_response,
)


class FirCorrectionTests(unittest.TestCase):
    def test_prototype_nyquist_limit_is_explicit(self) -> None:
        freq_hz = np.linspace(0.0, 200.0e6, 81)
        s21 = np.ones(freq_hz.size, dtype=complex)
        with self.assertRaises(ValueError):
            design_interpolated_ripple_correction_fir(
                freq_hz,
                s21,
                correction_band_hz=200.0e6,
                prototype_sample_rate_hz=300.0e6,
            )

    def test_interpolation_and_refinement_return_finite_2g_fir(self) -> None:
        freq_hz = np.linspace(0.0, 200.0e6, 81)
        ripple_db = 0.7 * np.sin(2.0 * np.pi * freq_hz / 80.0e6)
        phase = 0.08 * np.sin(2.0 * np.pi * freq_hz / 110.0e6)
        s21 = 10.0 ** (ripple_db / 20.0) * np.exp(1j * phase)
        design = design_interpolated_ripple_correction_fir(
            freq_hz,
            s21,
            correction_band_hz=200.0e6,
            prototype_sample_rate_hz=400.0e6,
            output_sample_rate_hz=800.0e6,
            prototype_tap_count=41,
        )

        self.assertEqual(design.refined_taps.size, 81)
        self.assertTrue(np.all(np.isfinite(design.refined_taps)))
        response = frequency_response(
            design.refined_taps,
            freq_hz,
            sample_rate_hz=800.0e6,
        )
        self.assertTrue(np.all(np.isfinite(response)))
        self.assertAlmostEqual(float(np.sum(design.refined_taps)), 1.0, places=6)

    def test_high_level_wrapper_returns_final_fir_sequence(self) -> None:
        freq_hz = np.linspace(0.0, 200.0e6, 81)
        ripple_db = 0.7 * np.sin(2.0 * np.pi * freq_hz / 80.0e6)
        phase = 0.08 * np.sin(2.0 * np.pi * freq_hz / 110.0e6)
        s21 = 10.0 ** (ripple_db / 20.0) * np.exp(1j * phase)

        fir = design_fir_sequence_from_s21(
            freq_hz,
            s21,
            correction_band_hz=200.0e6,
            output_sample_rate_hz=800.0e6,
            prototype_tap_count=41,
        )

        self.assertEqual(fir.size, 81)
        self.assertTrue(np.issubdtype(fir.dtype, np.floating))
        self.assertTrue(np.all(np.isfinite(fir)))
        self.assertAlmostEqual(float(np.sum(fir)), 1.0, places=6)

        explicit_design = design_interpolated_ripple_correction_fir(
            freq_hz,
            s21,
            correction_band_hz=200.0e6,
            prototype_sample_rate_hz=400.0e6,
            output_sample_rate_hz=800.0e6,
            prototype_tap_count=41,
        )
        np.testing.assert_allclose(fir, explicit_design.refined_taps)


if __name__ == "__main__":
    unittest.main()
