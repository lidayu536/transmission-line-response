from __future__ import annotations

import unittest
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

import numpy as np

from transmission_line_response import SimpleMultiexpFitResult, standing_wave_s21


class SParameterModelTests(unittest.TestCase):
    def test_simple_multiexp_fit_result_is_constructible(self) -> None:
        fit = SimpleMultiexpFitResult(
            alphas=np.array([0.25, 0.75]),
            taus=np.array([0.5, 2.0]),
            delay=1.0,
            sigma_alphas=np.array([0.01, 0.02]),
            sigma_taus=np.array([0.03, 0.04]),
            sigma_delay=0.05,
            covariance=np.eye(5),
        )
        self.assertAlmostEqual(fit.alpha_sum, 1.0)
        np.testing.assert_allclose(fit.normalized_alphas(), [0.25, 0.75])
        self.assertEqual(fit.s21(np.array([0.0, 0.1])).shape, (2,))

    def test_standing_wave_model_keeps_shape(self) -> None:
        freqs = np.linspace(0.0, 1.0, 11)
        s21 = standing_wave_s21(freqs, [1.0], [0.5], roundtrip=2.0, rho_mag=0.1)
        self.assertEqual(s21.shape, freqs.shape)


if __name__ == "__main__":
    unittest.main()