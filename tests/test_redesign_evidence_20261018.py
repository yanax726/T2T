"""Check patient-level display aggregation and flow denominators on synthetic data."""
import unittest
import numpy as np
import pandas as pd
from t2t_paper.cohort_flow_20261018 import exclusive_masks, membership_patterns
from t2t_paper.model_display_evidence_20261018 import ecdf, repeated_scores, paired_visits, confusion_rows, patient_bootstrap
from t2t_paper.balance_support_20261018 import effective_n
from t2t_paper.plot_source_aggregates_20261018 import combined_moments


class RedesignEvidenceTest(unittest.TestCase):
    def test_opposite_repeat_errors_do_not_cancel(self):
        y = np.array([2., 4.])
        predictions = np.array([[4., 6.], [0., 2.]])
        self.assertEqual(repeated_scores(y, predictions)[0], 2.)
        frame = pd.DataFrame(dict(patient_uid=['A', 'B', 'A', 'B'], repeat=[0, 0, 1, 1],
                                  observed=[2., 4., 2., 4.], prediction=[4., 6., 0., 2.]))
        values = ecdf(frame, np.array([1., 2.]), replicates=20)
        self.assertEqual(values.proportion_within.tolist(), [0., 1.])

    def test_pairs_require_same_patient_repeat_and_fold(self):
        frame = pd.DataFrame(dict(patient_uid=['A', 'A', 'A', 'A'], repeat=[0, 0, 1, 1],
                                  fold=[2, 2, 2, 3], visit_week=[0, 12, 0, 12],
                                  observed=[20., 5., 20., 5.], prediction=[18., 6., 17., 7.]))
        pair = paired_visits(frame, 0, 12)
        self.assertEqual(len(pair), 1)
        self.assertEqual(pair['repeat'].iloc[0], 0)

    def test_predicted_response_uses_its_own_baseline(self):
        pair = pd.DataFrame(dict(repeat=[0], observed_start=[20.], observed=[5.],
                                 prediction_start=[10.], prediction=[5.]))
        rows = pd.DataFrame(confusion_rows(pair, 0, 12))
        row = rows[rows.endpoint.eq('EASI75') & rows.observed_response.eq(1) & rows.predicted_response.eq(0)]
        self.assertEqual(int(row['count'].iloc[0]), 1)

    def test_nonpositive_baseline_is_excluded_and_counted(self):
        pair = pd.DataFrame(dict(repeat=[0, 0], observed_start=[20., 20.], observed=[5., 5.],
                                 prediction_start=[10., 0.], prediction=[5., 2.]))
        rows = pd.DataFrame(confusion_rows(pair, 0, 12))
        self.assertTrue(rows.n_excluded.eq(1).all())
        self.assertTrue(rows.n_evaluable.eq(1).all())
        self.assertTrue(rows.groupby('endpoint')['count'].sum().eq(1).all())

    def test_patient_bootstrap_retains_repeat_block(self):
        frame = pd.DataFrame(dict(patient_uid=['A', 'B', 'A', 'B'], repeat=[0, 0, 1, 1],
                                  observed=[2., 4., 2., 4.], prediction=[4., 6., 0., 2.]))
        result = {r['metric']: r for r in patient_bootstrap(frame, replicates=20)}
        for field in ['estimate', 'ci_low', 'ci_high']:
            self.assertEqual(result['MAE'][field], 2.)

    def test_visit_exclusion_does_not_necessarily_exclude_patient(self):
        frame = pd.DataFrame(dict(patient_uid=['A', 'A', 'B'], complete=[False, True, True]))
        kept, rows = exclusive_masks(frame, [('missing', lambda d: d.complete)])
        self.assertEqual(rows[0]['excluded_rows'], 1)
        self.assertEqual(rows[0]['patients_fully_lost'], 0)
        self.assertEqual(kept.patient_uid.nunique(), 2)

    def test_visit_patterns_partition_people_not_visits(self):
        frame = pd.DataFrame(dict(patient_uid=['A', 'A', 'B'], visit_week=[0, 24, 24]))
        patterns = membership_patterns(frame).set_index('pattern').n_patients.to_dict()
        self.assertEqual(patterns, {'W0+W24': 1, 'W24': 1})

    def test_effective_sample_size(self):
        self.assertEqual(effective_n([1., 1., 1.]), 3.)
        self.assertEqual(effective_n([1., 0., 0.]), 1.)
        self.assertTrue(np.isnan(effective_n([])))

    def test_public_group_moments_reproduce_pooled_sample(self):
        one, two = np.array([1., 2., 3.]), np.array([8.])
        frame = pd.DataFrame([dict(target='EASI', week=0, n_initial=len(x), n=len(x), n_missing=0,
                                   mean=x.mean(), sd=x.std(ddof=1) if len(x)>1 else np.nan)
                              for x in [one, two]])
        result = combined_moments(frame).iloc[0]
        pooled = np.r_[one, two]
        self.assertAlmostEqual(result['mean'], pooled.mean())
        self.assertAlmostEqual(result.sd, pooled.std(ddof=1))
        self.assertEqual(result.n, 4)


if __name__ == '__main__':
    unittest.main()
