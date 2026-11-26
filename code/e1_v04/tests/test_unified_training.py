import unittest
import numpy as np

from run_unified_vfa import fit_model, validate_layout_split, validation_choice
from unified_vfa_features import FEATURE_NAMES


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


if __name__ == '__main__':
    unittest.main()
