"""Public calculation checks; no participant data are needed."""
import json
import math
import os
import unittest
from pathlib import Path

from cytot2t.curation.unified_bsa_three_calculator import calculate

ROOT = Path(os.environ.get("T2T_ROOT", Path(__file__).resolve().parents[1]))


class UnifiedScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads((ROOT / "config/unified_bsa_three_v1_20261017.json").read_text())
        cls.row = {"bsa": 10.0, "log2_CCL17": 10.0, "log2_CCL26": 7.0, "log2_IL-5": 1.0}

    def test_raw_and_log_inputs_agree(self):
        raw = {"bsa": 10.0, "CCL17": 1024.0, "CCL26": 128.0, "IL-5": 2.0}
        self.assertAlmostEqual(calculate(raw, self.artifact, "raw"), calculate(self.row, self.artifact), places=12)

    def test_absent_lesion_area_is_zero(self):
        self.assertEqual(calculate(dict(self.row, bsa=0.0), self.artifact), 0.0)

    def test_no_stage_or_drug_required(self):
        value = calculate(self.row, self.artifact)
        for stage, treatment in [(0, "none"), (1, "dupilumab"), (1, "abrocitinib")]:
            self.assertEqual(value, calculate(dict(self.row, stage=stage, treatment=treatment), self.artifact))

    def test_reject_invalid_area(self):
        for value in [-1.0, 100.01, math.inf, math.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate(dict(self.row, bsa=value), self.artifact)

    def test_reject_nonpositive_raw_concentrations(self):
        for value in [0.0, -1.0, math.inf, math.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate({"bsa": 10.0, "CCL17": value, "CCL26": 128.0, "IL-5": 2.0}, self.artifact, "raw")

    def test_missing_values_are_not_imputed(self):
        incomplete = dict(self.row)
        del incomplete["log2_IL-5"]
        with self.assertRaises(KeyError):
            calculate(incomplete, self.artifact)
        for value in [math.nan, math.inf, ""]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate(dict(self.row, **{"log2_IL-5": value}), self.artifact)

    def test_output_is_bounded(self):
        high = dict(self.row, bsa=100.0, log2_CCL17=10000)
        low = dict(self.row, log2_CCL17=-10000)
        self.assertEqual(calculate(high, self.artifact), 72.0)
        self.assertEqual(calculate(low, self.artifact), 0.0)


if __name__ == "__main__":
    unittest.main()
