"""A service master can admit two new pickups absent from every source plan."""
from copy import deepcopy
import pickle
import unittest

from e1_sim.online import evaluate_plan, pickup_variants
from e1_sim.quantities import dec
from service_action_mip import service_action_pool
from test_unified_vfa_policy import two_trucks


class ServiceMipTests(unittest.TestCase):
    def setUp(self):
        try:
            import gurobipy
        except ImportError:
            self.skipTest('requires the optional Gurobi backend')
        self.known, plan = two_trucks()
        for i in (1, 2):
            oid = f'P{i}'
            station = f'L{i}+'
            self.known.orders[oid] = dict(id=oid, kind='P', weight=1, accepted=None,
                city=f'C{i}', xy=self.known.network['stations'][station], release=dec(0),
                deadline=dec(100), completed_at=None, collected_at=None, available_at=None,
                assigned_drone=None, station=None, eligible_stations=[station])
            self.known._new_lot(oid, 1, 'customer', oid)
        plan['decisions'] = {'P1': False, 'P2': False}
        self.bases = [evaluate_plan(self.known, plan)]
        for i in (1, 2):
            oid = f'P{i}'
            one = deepcopy(plan)
            one.update(id=f'source_{oid}', decisions={order: order == oid for order in one['decisions']})
            for action in pickup_variants(self.known, one, oid):
                if action['new_trucks'] or not any(a['kind'] == 'truck_load' and a['actor'] == f'T{i}'
                                                     for a in action['actions']):
                    continue
                checked = evaluate_plan(self.known, action)
                if checked['feasible'] and checked['allowed']:
                    self.bases.append(checked)
                    break
        self.assertEqual(len(self.bases), 3)

    def test_combine_independent_service_chains(self):
        before = pickle.dumps(self.known)
        pool, stats = service_action_pool(self.known, self.bases, pattern_limit=None)
        self.assertFalse(stats['truncated'])
        self.assertEqual(stats['status'], 'restricted_service_patterns_exhausted')
        self.assertEqual(stats['attempted'], 1)
        self.assertEqual(stats['accepted'], 1)
        self.assertEqual(len(pool), 4)
        both = next(r for r in pool if all(r['plan']['decisions'].values()))
        self.assertTrue(both['state'].is_terminal())
        self.assertEqual({both['state'].orders[p]['station'] for p in ('P1', 'P2')}, {'L1+', 'L2+'})
        self.assertEqual(before, pickle.dumps(self.known))
        self.assertEqual(pool[:3], self.bases)

    def test_cap_retains_original_feasible_incumbents(self):
        delayed = deepcopy(self.bases[0]['plan'])
        delayed['id'] = 'delayed_T1'
        delayed['actions'][0]['at'] = dec(1)
        self.bases.append(evaluate_plan(self.known, delayed))
        self.assertTrue(self.bases[-1]['feasible'])
        pool, stats = service_action_pool(self.known, self.bases, pattern_limit=1)
        self.assertEqual(stats['attempted'], 1)
        self.assertTrue(stats['truncated'])
        self.assertEqual(pool[:4], self.bases)
        self.assertTrue(all(row['feasible'] for row in pool))


if __name__ == '__main__':
    unittest.main()
