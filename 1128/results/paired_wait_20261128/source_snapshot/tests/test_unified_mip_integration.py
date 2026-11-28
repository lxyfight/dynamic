"""Policy-level parity between exhaustive tiny neighborhoods and the MIP backend."""
from argparse import ArgumentTypeError
from importlib.util import find_spec
import unittest
from unittest.mock import patch

from e1_sim.online import evaluate_plan
from test_unified_vfa_policy import two_trucks, model
from unified_vfa_features import FEATURE_NAMES
from unified_vfa_policy import UnifiedVFAPolicy, parse_pattern_limit


class ActionBackendTests(unittest.TestCase):
    def test_search_configuration(self):
        self.assertIsNone(parse_pattern_limit('all'))
        self.assertEqual(parse_pattern_limit('3'), 3)
        with self.assertRaises(ArgumentTypeError):
            parse_pattern_limit('0')
        for options in (dict(action_solver='unknown'), dict(mip_pattern_limit=0),
                        dict(mip_pattern_limit=True)):
            with self.assertRaises(ValueError):
                UnifiedVFAPolicy(**options)

    @unittest.skipUnless(find_spec('gurobipy'), 'optional Gurobi installation required')
    def test_joint_wait_profit_uses_physically_verified_nonadditive_feature(self):
        known, plan = two_trucks()
        base = evaluate_plan(known, plan)

        def future(state, row, h):
            both_wait = len(row['plan'].get('wait_reviews', {})) == 2
            return (1000. if both_wait else 0.,) + (0.,) * (len(FEATURE_NAMES) - 1)

        policies = [UnifiedVFAPolicy(model=model(), enhancement_budget=0,
                                     action_solver=solver, mip_pattern_limit=16)
                    for solver in ('bounded', 'gurobi')]
        with patch('unified_vfa_policy.features', side_effect=future):
            actual = [policy.choose(known, [base]) for policy in policies]
        self.assertEqual(actual[0]['id'], actual[1]['id'])
        self.assertEqual(len(actual[1]['plan']['wait_reviews']), 2)
        mip = policies[1]
        event = mip.selection_log[-1]
        self.assertFalse(event['truncated'])
        self.assertEqual(event['waiting']['winner_index'], next(
            i for i, row in enumerate(event['pool']) if row['id'] == actual[1]['id']))
        self.assertEqual(mip.search_metrics()['wait_patterns_accepted'], 3)


if __name__ == '__main__':
    unittest.main()
