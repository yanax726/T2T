"""Numerical checks for display precision; no participant data are required."""
from copy import deepcopy
import unittest
import numpy as np
import pandas as pd
from cytot2t.curation.rounding_20261017 import rounded_formula
from cytot2t.curation.unified_model_20261017 import evaluate_formula


class RoundingAuditTest(unittest.TestCase):
    def setUp(self):
        self.formula = dict(intercept=0., bsa_factor=.5127215790257503,
                            bsa_log2_coefficients={'log2_CCL17':.009847079991894058,
                                                  'log2_CCL26':.009646929477400354,
                                                  'log2_IL-5':-.008270558237379214})
        self.frame = pd.DataFrame({'bsa':[0.,10.,79.], 'log2_CCL17':[10.,10.,15.],
                                   'log2_CCL26':[7.,7.,12.], 'log2_IL-5':[1.,1.,-6.]})

    def test_scaled_display_means_divide_by_100_once(self):
        formula=rounded_formula(self.formula,'scaled100_2dp')
        expected=self.frame.bsa.to_numpy()/100*(51.27+.98*self.frame.log2_CCL17.to_numpy()+.96*self.frame.log2_CCL26.to_numpy()-.83*self.frame['log2_IL-5'].to_numpy())
        np.testing.assert_allclose(evaluate_formula(self.frame,formula),np.clip(expected,0,72),rtol=0,atol=1e-12)

    def test_scaled_two_decimals_preserve_four_effective_decimals(self):
        np.testing.assert_allclose(evaluate_formula(self.frame,rounded_formula(self.formula,'scaled100_2dp')),
                                   evaluate_formula(self.frame,rounded_formula(self.formula,'direct_4dp')),rtol=0,atol=1e-12)

    def test_direct_two_decimals_is_different_policy(self):
        formula=rounded_formula(self.formula,'direct_2dp')
        self.assertEqual(formula['bsa_factor'],.51)
        self.assertEqual(formula['bsa_log2_coefficients'],{'log2_CCL17':.01,'log2_CCL26':.01,'log2_IL-5':-.01})
        self.assertGreater(abs(evaluate_formula(self.frame,formula)[2]-evaluate_formula(self.frame,rounded_formula(self.formula,'scaled100_2dp'))[2]),.1)

    def test_frozen_formula_not_modified(self):
        before=deepcopy(self.formula)
        rounded_formula(self.formula,'direct_2dp')
        self.assertEqual(self.formula,before)
        self.assertEqual(rounded_formula(self.formula,'full_precision'),before)

    def test_zero_anchor_and_output_bounds_preserved(self):
        self.assertEqual(evaluate_formula(self.frame,rounded_formula(self.formula,'scaled100_2dp'))[0],0.)
        high=self.frame.iloc[[1]].copy();high['bsa']=100.;high['log2_CCL17']=10000.
        self.assertEqual(evaluate_formula(high,rounded_formula(self.formula,'scaled100_2dp'))[0],72.)


if __name__=='__main__':
    unittest.main()
