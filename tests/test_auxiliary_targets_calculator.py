"""Calculation checks for the three separately fitted recorded targets."""
import json
import math
import os
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from cytot2t.curation.auxiliary_targets_calculator_20261017 import calculate
from cytot2t.curation.auxiliary_targets_20261017 import formula_predictions

ROOT=Path(os.environ.get("T2T_ROOT",Path(__file__).resolve().parents[1]))


class AuxiliaryCalculatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts={target:json.loads((ROOT/f"results/tables/update_20261017/model/auxiliary_targets/formula_{target}.json").read_text()) for target in ["pp_nrs","dlqi","iga"]}
        cls.row={"age":35,"bsa":10,"log2_CCL17":10,"log2_CCL26":7,"log2_IL-5":1}

    def test_raw_and_log_inputs_match(self):
        raw={"age":35,"bsa":10,"CCL17":1024,"CCL26":128,"IL-5":2}
        for target,artifact in self.artifacts.items():
            with self.subTest(target=target):
                self.assertEqual(calculate(raw,artifact,"raw"),calculate(self.row,artifact))

    def test_scalar_and_vector_formula_reconstruction(self):
        for target,artifact in self.artifacts.items():
            value=calculate(self.row,artifact)
            expected=formula_predictions(pd.DataFrame([self.row]),artifact)
            if target=="iga":
                np.testing.assert_allclose([value[f"p{i}"] for i in range(6)],expected[0],atol=1e-12)
            else:
                self.assertAlmostEqual(value[target+"_estimate"],expected[0],places=12)

    def test_dlqi_rejects_age17(self):
        with self.assertRaises(ValueError):
            calculate(dict(self.row,age=17),self.artifacts["dlqi"])

    def test_no_phase_or_drug_dispatch(self):
        for artifact in self.artifacts.values():
            self.assertEqual(calculate(self.row,artifact),calculate(dict(self.row,stage=1,treatment="any"),artifact))

    def test_invalid_values_rejected(self):
        for artifact in self.artifacts.values():
            for value in [-1,101,math.nan,math.inf]:
                with self.assertRaises(ValueError):calculate(dict(self.row,bsa=value),artifact)
            with self.assertRaises(ValueError):calculate(dict(self.row,**{"log2_IL-5":math.nan}),artifact)
            with self.assertRaises(ValueError):calculate({"age":35,"bsa":10,"CCL17":0,"CCL26":128,"IL-5":2},artifact,"raw")

    def test_ordinal_categories_are_probabilities(self):
        result=calculate(self.row,self.artifacts["iga"])
        probabilities=[result[f"p{i}"] for i in range(6)]
        self.assertAlmostEqual(sum(probabilities),1)
        self.assertTrue(all(p>=0 for p in probabilities))
        self.assertIn(result["predicted_recorded_grade"],range(6))


if __name__=="__main__":unittest.main()
