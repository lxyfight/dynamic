"""Persist and check the pickup deadline distribution and station processing rules."""
import unittest
from copy import deepcopy

from run_unified_vfa import audited_episode, fit_model, resolve_modes
from uniform_wait_worlds import make_world
from unified_vfa_features import FEATURE_NAMES
from unified_vfa_policy import UnifiedVFAPolicy


class UnifiedOperatingRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = make_world(710, 0, 4, .5, 'uniform', history_days=1,
                               improvement_budget=0, arrival_mode='p2')
        cls.model = fit_model([[dict(x=[1.] * len(FEATURE_NAMES), y=2.)]],
                              5, arrival_mode='p2', review_intervals=[2, 5, 10])

    def test_current_weights_replay_with_their_world(self):
        rules = self.model['operating_rules']
        self.assertEqual(rules['pickup_deadline_lags'], [240, 300, 360])
        self.assertEqual(rules['drone_handling_setup_per_request'], 0)
        self.assertEqual(rules['truck_handling_setup'], 0)
        self.assertEqual(rules['customer_service_per_request'], 1)
        row, _ = audited_episode(self.world, 5, self.model, review_intervals=[2, 5, 10],
                                 enhancement_budget=0)
        self.assertTrue(row['trained_model'])

    def test_retraining_required_for_previous_saved_weights(self):
        old = dict(self.model, version='unified-vfa-development-v2')
        old.pop('operating_rules')
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            resolve_modes(None, None, default_mode='adaptive', model=old)
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            UnifiedVFAPolicy(model=old, review_intervals=[2, 5, 10])

    def test_weights_with_different_physical_or_demand_assumptions_are_rejected(self):
        for name, other in [('pickup_deadline_lags', [360]),
                            ('drone_handling_setup_per_request', 1)]:
            with self.subTest(name=name):
                outside = deepcopy(self.model)
                outside['operating_rules'][name] = other
                with self.assertRaisesRegex(ValueError, 'different pickup deadline or handling'):
                    audited_episode(self.world, 5, outside, review_intervals=[2, 5, 10])
