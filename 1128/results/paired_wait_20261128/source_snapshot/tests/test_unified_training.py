import unittest
import numpy as np

from run_unified_vfa import fit_model, validate_layout_split, validation_choice, validation_residual_fit
from unified_vfa_features import FEATURE_NAMES, active_feature_names


class UnifiedTrainingTests(unittest.TestCase):
    def test_h_selection_uses_validation_and_has_stable_tie_break(self):
        rows = [dict(h=10, validation_mean_net=4), dict(h=5, validation_mean_net=4),
                dict(h=2, validation_mean_net=3)]
        self.assertEqual(validation_choice(rows)['h'], 5)
        with self.assertRaises(ValueError):
            validate_layout_split([710], [710, 810])

    def test_equal_day_fit_is_invariant_to_duplicate_samples_within_day(self):
        n = len(FEATURE_NAMES)
        a = dict(x=[1.] * n, y=12.)
        b = dict(x=list(np.linspace(0.1, 1.2, n)), y=-7.)
        first = fit_model([[a], [b]], 5)
        second = fit_model([[a] * 8, [b]], 5)
        np.testing.assert_allclose(first['weights'], second['weights'])
        np.testing.assert_allclose(first['scales'], second['scales'])
        self.assertEqual(first['h'], 5)
        self.assertEqual(first['feature_names'], list(FEATURE_NAMES))
        self.assertEqual(first['fit']['target'], 'audited_full_day_net_minus_J_known')

    def test_fit_rejects_missing_trajectories_and_nonfinite_labels(self):
        with self.assertRaises(ValueError):
            fit_model([[]], 5)
        with self.assertRaises(ValueError):
            fit_model([[dict(x=[0.] * len(FEATURE_NAMES), y=float('nan'))]], 5)

    def test_ablations_refit_selected_columns_in_the_original_schema(self):
        days = [[dict(x=[2., 4., 6., 1., 7., 3., 8.], y=7.)]]
        model = fit_model(days, 5, ridge=1, feature_set='added_trip_reliance')
        self.assertEqual(model['active_features'], [FEATURE_NAMES[2]])
        np.testing.assert_allclose(model['scales'], [1, 1, 6, 1, 1, 1, 1])
        np.testing.assert_allclose(model['weights'], [0, 0, 3.5, 0, 0, 0, 0])
        self.assertNotEqual(model['weights'][2], fit_model(days, 5)['weights'][2])
        zero = fit_model(days, 5, feature_set='none')
        self.assertEqual(zero['weights'], [0.] * len(FEATURE_NAMES))
        self.assertEqual(active_feature_names('added_trip_reliance,existing_chain_opportunity'),
                         FEATURE_NAMES[:3])
        with self.assertRaises(ValueError):
            active_feature_names('added_trip_reliance,added_trip_reliance')

    def test_test_layouts_are_unused_in_validation_and_ties_prefer_simpler_models(self):
        with self.assertRaises(ValueError):
            validate_layout_split([1], [2], [2, 3])
        with self.assertRaises(ValueError):
            validate_layout_split([1], [2], [3, 1])
        validate_layout_split([1], [2], [3])
        candidates = [dict(h=2, feature_set='full', validation_mean_net=15),
                      dict(h=5, feature_set='added_trip_reliance', validation_mean_net=15),
                      dict(h=2, feature_set='none', validation_mean_net=14)]
        self.assertEqual(validation_choice(candidates)['feature_set'], 'added_trip_reliance')

    def test_heldout_residual_diagnostics_weight_days_equally(self):
        model = dict(weights=[2., 0.], scales=[2., 1.])
        short = [dict(x=[1., 0.], y=0.)]  # error squared = 1
        long = [dict(x=[1., 0.], y=4.)] * 3  # error squared = 9
        fit = validation_residual_fit([short, [], long], model)
        self.assertEqual((fit['days'], fit['samples']), (2, 4))
        self.assertAlmostEqual(fit['rmse'], np.sqrt(5))
        self.assertAlmostEqual(fit['r2'], 1 - 5 / 4)
        self.assertIsNone(validation_residual_fit([short], model)['r2'])
        self.assertIsNone(validation_residual_fit([[]], model)['rmse'])


if __name__ == '__main__':
    unittest.main()
