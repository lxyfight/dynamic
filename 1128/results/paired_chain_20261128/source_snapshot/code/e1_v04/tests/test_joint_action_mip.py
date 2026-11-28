"""Exact-versus-exhaustive and truncated-search diagnostics for online MIP."""
from copy import deepcopy
from itertools import product
import unittest
from unittest.mock import patch

from e1_sim.online import evaluate_plan, install_plan, remaining_plan
from e1_sim.quantities import dec
from joint_action_mip import search_joint_actions
from test_unified_vfa_policy import two_trucks
from unified_vfa_policy import joint_wait_pool, replay_wait_combination

try:
    import gurobipy as gp
except ImportError:
    gp = None


@unittest.skipIf(gp is None, 'optional gurobipy dependency')
class JointActionMipTests(unittest.TestCase):
    def test_adaptive_trucks_choose_independent_intervals(self):
        known, plan = two_trucks()
        base = evaluate_plan(known, plan)
        menu = (2, 5, 10)

        def timers(row):
            return tuple(row['plan']['wait_reviews'].get(tid, 0) for tid in ('T1', 'T2'))

        def price(row):
            return float(row['score']) + (1000. if timers(row) == (2, 10) else 0.)

        pool, audit = search_joint_actions(known, [base], 5, review_intervals=menu,
                                           pattern_limit=None, score=price)
        self.assertEqual({timers(row) for row in pool}, set(product((0, *menu), repeat=2)))
        self.assertEqual(len(pool), 16)
        self.assertEqual(audit['by_base'][0]['trucks'], ['T1', 'T2'])
        self.assertEqual(audit['attempted'], 15)
        self.assertFalse(audit['truncated'])
        self.assertEqual(timers(pool[audit['winner_index']]), (2, 10))
        limited, log = search_joint_actions(known, [base], 5, review_intervals=menu,
                                            pattern_limit=4, score=price)
        self.assertTrue(log['truncated'])
        self.assertEqual(log['possible_patterns'], 15)
        self.assertEqual(len({timers(row) for row in limited}), 5)
        self.assertEqual(log['verified_objective'], max(map(price, limited)))

    def test_unexpired_timer_offers_only_keep_or_leave(self):
        known, plan = two_trucks()
        plan['actions'][0]['at'] = 5
        plan['wait_reviews']['T1'] = dec(5)
        install_plan(known, plan)
        known._advance(dec(2))
        plan = remaining_plan(known)
        plan['actions'].append(dict(at=2, kind='truck_continue_circuit', actor='T2', station='L2+'))
        pool, audit = search_joint_actions(known, [evaluate_plan(known, plan)], 5,
                                           review_intervals=(2, 5, 10), pattern_limit=None,
                                           score=lambda row: float(row['score']))
        deadlines = [r['plan']['wait_reviews'] for r in pool]
        self.assertEqual(audit['possible_patterns'], 7)
        self.assertEqual({(x.get('T1'), x.get('T2')) for x in deadlines},
                         set(product((None, 5), (None, 4, 7, 12))))

    def test_all_coalitions_match_exhaustive_joint_search(self):
        known, plan = two_trucks()
        second = deepcopy(plan)
        second['id'] = 'second'
        bases = [evaluate_plan(known, p) for p in (plan, second)]

        def price(row):
            # Single-move prices cannot predict the joint improvement on the
            # less-promising base; only complete replayed scores can do that.
            b = row['id'].startswith('second')
            n = len(row['plan']['wait_reviews'])
            return float(row['score']) - 2*b + (100. if b and n == 2 else 0.)

        expected, _ = joint_wait_pool(known, bases, 5, budget_per_base=3)
        actual, audit = search_joint_actions(known, bases, 5, pattern_limit=100, score=price)
        self.assertEqual({r['id'] for r in actual}, {r['id'] for r in expected})
        self.assertEqual(audit['possible_patterns'], 6)
        self.assertEqual(audit['attempted'], 6)
        self.assertEqual(audit['verified_objective'], max(map(price, expected)))
        self.assertTrue(audit['complete_given_bases'])
        self.assertFalse(audit['truncated'])
        winner = actual[audit['winner_index']]
        self.assertTrue(winner['id'].startswith('second'))
        self.assertEqual(len(winner['plan']['wait_reviews']), 2)

    def test_failed_pattern_does_not_cut_its_supersets(self):
        known, plan = two_trucks()
        bases = [evaluate_plan(known, plan)]

        def check(state, row, h, options, selected):
            result, info = replay_wait_combination(state, row, h, options, selected)
            if selected == (0,):
                result = dict(feasible=False, allowed=False, reason='single infeasible')
                info.update(result)
            return result, info

        with patch('unified_vfa_policy.replay_wait_combination', side_effect=check):
            rows, audit = search_joint_actions(known, bases, 5, pattern_limit=None,
                score=lambda r: len(r['plan']['wait_reviews']))
        self.assertEqual(audit['infeasible_patterns'], 1)
        self.assertEqual(len(rows), 3)
        self.assertTrue(audit['complete_given_bases'])
        self.assertEqual(audit['verified_objective'], 2.)

    def test_limit_keeps_all_bases_and_stable_equal_value_tie(self):
        known, plan = two_trucks()
        second = deepcopy(plan)
        second['id'] = 'a_base'
        bases = [evaluate_plan(known, p) for p in (plan, second)]
        pool, audit = search_joint_actions(known, bases, 5, pattern_limit=1, score=lambda row: 1.)
        self.assertTrue(audit['truncated'])
        self.assertFalse(audit['complete_given_bases'])
        self.assertEqual(audit['attempted'], 1)
        self.assertEqual(pool[:2], bases)
        self.assertEqual(audit['winner_id'], 'a_base')

    def test_closed_epoch_has_only_base_and_invalid_limits_fail(self):
        known, plan = two_trucks()
        known.public['revelation_closed'] = True
        base = evaluate_plan(known, plan)
        rows, stats = search_joint_actions(known, [base], 5, pattern_limit=1, score=lambda r: -1.)
        self.assertEqual(rows, [base])
        self.assertEqual(stats['attempted'], 0)
        self.assertTrue(stats['complete_given_bases'])
        for limit in (-1, 0, True):
            with self.assertRaises(ValueError):
                search_joint_actions(known, [base], 5, pattern_limit=limit, score=lambda r: 0.)


if __name__ == '__main__':
    unittest.main()
