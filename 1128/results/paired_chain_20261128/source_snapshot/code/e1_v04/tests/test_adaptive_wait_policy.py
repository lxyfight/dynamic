"""Adaptive truck-local review clocks and common-horizon VFA scoring."""
import unittest
from unittest.mock import patch

from e1_sim.online import evaluate_plan, install_plan, remaining_plan
from e1_sim.quantities import dec
from test_unified_vfa_policy import two_trucks, model
from unified_vfa_features import FEATURE_NAMES, features
from unified_vfa_policy import UnifiedVFAPolicy, joint_wait_pool


MENU = [2, 5, 10]


def all_actions(state, plan):
    base = evaluate_plan(state, plan)
    return joint_wait_pool(state, [base], 5, budget_per_base=30, review_intervals=MENU)


class AdaptiveReviewTests(unittest.TestCase):
    def test_exhaustive_compound_timers_and_nonconstant_features(self):
        known, plan = two_trucks()
        pool, stats = all_actions(known, plan)
        timers = [(r['plan']['wait_reviews'].get('T1', 0),
                   r['plan']['wait_reviews'].get('T2', 0)) for r in pool]
        self.assertEqual(set(timers), {(a, b) for a in [0] + MENU for b in [0] + MENU})
        self.assertEqual(len(timers), 16)
        self.assertEqual(stats['attempted'], 15)
        self.assertFalse(stats['truncated'])
        self.assertLess(features(known, pool[timers.index((2, 0))], 5)[0],
                        features(known, pool[timers.index((10, 0))], 5)[0])
        actions, stats = joint_wait_pool(known, [pool[0]], 5, budget_per_base=0, review_intervals=MENU)
        self.assertEqual(actions, [pool[0]])
        self.assertTrue(stats['truncated'])

    def test_only_expired_truck_selects_fresh_interval(self):
        known, plan = two_trucks()
        rows, _ = all_actions(known, plan)
        initial = next(r for r in rows if r['plan']['wait_reviews'] == {'T1': 2, 'T2': 10})
        install_plan(known, initial['plan'])
        known._advance(dec(1))
        early, _ = all_actions(known, remaining_plan(known))
        self.assertTrue(all(r['plan']['wait_reviews'].get('T1', 2) == 2 and
                            r['plan']['wait_reviews'].get('T2', 10) == 10 for r in early))
        known._advance(dec(2))
        renewed, _ = all_actions(known, remaining_plan(known))
        self.assertTrue(any(r['plan']['wait_reviews'] == {'T1': 7, 'T2': 10} for r in renewed))
        self.assertTrue(all(r['plan']['wait_reviews'].get('T2', 10) == 10 for r in renewed))

    def test_adaptive_model_compatibility_and_feature_horizon(self):
        adaptive = dict(model(), wait_mode='adaptive', review_intervals=MENU, feature_horizon=5)
        for options in (dict(model=adaptive), dict(model=model(), review_intervals=MENU),
                        dict(model=adaptive, review_intervals=[2, 10]),
                        dict(model=adaptive, review_intervals=MENU, feature_horizon=4),
                        dict(review_intervals=[0, 5]), dict(review_intervals=[2, 2])):
            with self.assertRaises(ValueError):
                UnifiedVFAPolicy(**options)
        self.assertEqual(UnifiedVFAPolicy(model=adaptive, review_intervals=MENU).feature_horizon, 5)
        self.assertEqual(UnifiedVFAPolicy(h=2, feature_horizon=5).feature_horizon, 5)

    def test_one_scorer_selects_different_timers_at_same_event(self):
        known, plan = two_trucks()
        policy = UnifiedVFAPolicy(model=dict(model(weight=100), wait_mode='adaptive',
            review_intervals=MENU), review_intervals=MENU, enhancement_budget=0,
            wait_budget_per_base=20)
        def future(state, row, horizon):
            self.assertEqual(horizon, 5)
            desired = row['plan']['wait_reviews'] == {'T1': 2, 'T2': 10}
            return (float(desired),) + (0.,)*(len(FEATURE_NAMES)-1)
        with patch('unified_vfa_policy.features', side_effect=future):
            chosen = policy.choose(known, [evaluate_plan(known, plan)])
        self.assertEqual(chosen['plan']['wait_reviews'], {'T1': 2, 'T2': 10})
        self.assertEqual(policy.search_metrics()['chosen_wait_interval_counts'], {'2': 1, '10': 1})
