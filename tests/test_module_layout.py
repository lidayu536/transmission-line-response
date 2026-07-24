from __future__ import annotations

import unittest
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


class ModuleLayoutTests(unittest.TestCase):
    def test_preferred_subpackage_imports(self) -> None:
        from transmission_line_response.analysis import analyze_phase_group_delay, analyze_rise_from_s21
        from transmission_line_response.io import read_s21, read_touchstone_2port
        from transmission_line_response.sparams import prepare_s21, s21_to_impulse_response

        self.assertTrue(callable(analyze_phase_group_delay))
        self.assertTrue(callable(analyze_rise_from_s21))
        self.assertTrue(callable(read_s21))
        self.assertTrue(callable(read_touchstone_2port))
        self.assertTrue(callable(prepare_s21))
        self.assertTrue(callable(s21_to_impulse_response))

    def test_compatibility_facades(self) -> None:
        from transmission_line_response.phase import analyze_phase_group_delay
        from transmission_line_response.s21 import s21_to_impulse_response
        from transmission_line_response.time_domain import analyze_rise_from_s21

        self.assertTrue(callable(analyze_phase_group_delay))
        self.assertTrue(callable(analyze_rise_from_s21))
        self.assertTrue(callable(s21_to_impulse_response))


if __name__ == "__main__":
    unittest.main()
