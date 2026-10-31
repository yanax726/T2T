"""Boundary and repeated-patient checks for classification uncertainty."""
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.stats import beta

from t2t_paper.model_display_evidence_20261018 import response_uncertainty


def pairs(observed, predicted_repeats):
    rows = []
    for repeat, predicted in enumerate(predicted_repeats):
        for patient, (y, p) in enumerate(zip(observed, predicted)):
            rows.append(dict(patient_uid=f'p{patient}', repeat=repeat, fold=patient % 2,
                             observed_start=20., observed=y,
                             prediction_start=20., prediction=p))
    return pd.DataFrame(rows)


class ResponseIntervalTest(unittest.TestCase):
    def test_all_success_intervals_are_nonzero_and_do_not_double_patients(self):
        # Seven responders and seven nonresponders, perfectly classified twice.
        observed = [1.] * 7 + [19.] * 7
        average, individual = response_uncertainty(
            pairs(observed, [observed, observed]), 'W0_W24', replicates=30, return_per_repeat=True)
        self.assertEqual(len(average), 9)
        self.assertEqual(len(individual), 18)
        for row in average:
            n = 14 if row['metric'] == 'agreement' else 7
            self.assertEqual(row['denominator_min'], n)
            self.assertEqual(row['denominator_max'], n)
            self.assertEqual(row['estimate'], 1.)
            # 97.5% two-sided CP with all successes has analytic lower bound.
            self.assertAlmostEqual(row['ci_low'], .0125 ** (1 / n), places=11)
            self.assertEqual(row['ci_high'], 1.)
            self.assertLess(row['ci_low'], .0125 ** (1 / (2 * n)))
            self.assertEqual(row['bootstrap_diagnostic_ci_low'], 1.)
            self.assertEqual(row['bootstrap_diagnostic_ci_high'], 1.)
            self.assertEqual(row['bootstrap_interval_role'], 'diagnostic_only')
        for row in individual:
            self.assertLess(row['wilson_ci_low'], row['wilson_ci_high'])
            self.assertEqual(row['bonferroni_cp_confidence_level'], .975)

    def test_all_failure_intervals_have_positive_width(self):
        observed = [1.] * 7 + [19.] * 7
        predictions = [19.] * 7 + [1.] * 7
        rows = response_uncertainty(pairs(observed, [predictions, predictions]), 'W0_W24', replicates=30)
        for row in rows:
            n = row['denominator_min']
            self.assertEqual(row['estimate'], 0.)
            self.assertEqual(row['ci_low'], 0.)
            self.assertAlmostEqual(row['ci_high'], 1 - .0125 ** (1 / n), places=11)

    def test_score_each_repeat_and_average_bounds_separately(self):
        # At EASI75, mean predicted score 6 misses the response, but mean
        # repetition-specific sensitivity is 1/2. Preserve the latter estimand.
        frame = pairs([4., 15.], [[4., 15.], [8., 15.]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            average, individual = response_uncertainty(frame, 'W0_W12', replicates=30,
                                                       return_per_repeat=True)
        selected = [r for r in average if r['endpoint'] == 'EASI75' and r['metric'] == 'sensitivity'][0]
        self.assertEqual(selected['estimate'], .5)
        expected_low = (.0125 + 0.) / 2
        expected_high = (1. + .9875) / 2
        self.assertAlmostEqual(selected['ci_low'], expected_low)
        self.assertAlmostEqual(selected['ci_high'], expected_high)
        self.assertEqual(selected['denominator_min'], 1)
        self.assertEqual(selected['n_patients'], 2)

    def test_mixed_counts_match_independent_beta_quantiles(self):
        observed = [4.] * 10 + [15.] * 10
        predictions = [[4.] * 7 + [8.] * 3 + [15.] * 9 + [4.],
                       [4.] * 8 + [8.] * 2 + [15.] * 8 + [4.] * 2]
        average = response_uncertainty(pairs(observed, predictions), 'W0_W12', replicates=30)
        row = [r for r in average if r['endpoint'] == 'EASI75' and r['metric'] == 'sensitivity'][0]
        self.assertAlmostEqual(row['estimate'], .75)
        self.assertAlmostEqual(row['ci_low'], np.mean([beta.ppf(.0125, k, 11 - k) for k in [7, 8]]))
        self.assertAlmostEqual(row['ci_high'], np.mean([beta.ppf(.9875, k + 1, 10 - k) for k in [7, 8]]))

    def test_absent_observed_class_is_not_an_estimated_zero(self):
        observed = [1.] * 7
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            average, individual = response_uncertainty(
                pairs(observed, [observed, observed]), 'W0_W24', replicates=30, return_per_repeat=True)
        for row in average:
            if row['metric'] == 'specificity':
                self.assertTrue(np.isnan(row['estimate']))
                self.assertTrue(np.isnan(row['ci_low']))
                self.assertTrue(np.isnan(row['ci_high']))
                self.assertEqual(row['n_estimable_repeats'], 0)
                self.assertEqual(row['interval_status'], 'unavailable_repeat_class')
        for row in individual:
            if row['metric'] == 'specificity':
                self.assertEqual(row['denominator'], 0)
                self.assertTrue(np.isnan(row['wilson_ci_low']))
                self.assertEqual(row['interval_status'], 'no_observed_class')


if __name__ == '__main__':
    unittest.main()
