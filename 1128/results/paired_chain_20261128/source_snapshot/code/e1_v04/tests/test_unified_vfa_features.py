"""Information boundary and interpretation checks for candidate VFA features."""
from copy import deepcopy
from math import isfinite
from types import SimpleNamespace
import unittest

from e1_sim.unknown_demand import public_law
from unified_vfa_features import FEATURE_GROUPS, FEATURE_NAMES, FEATURE_UNITS, features


def fixture():
    law = public_law(1, 10, end=10)
    law['demand_model']['attributes'].update(weight_min=10, weight_max=10, pickup_deadline_lag=10)
    law['demand_model']['spatial_service_lags_by_city'] = {'C1':[2., 4.]}
    known = SimpleNamespace(now=0, public=law,
        cfg=dict(truck_capacity=100, truck_handling_setup=0, truck_handling_per_kg=0),
        network=dict(stations={'L1+':[0,1], 'L1-':[0,-1]}),
        drones={'U+':dict(home='L1+', active_operation=None),
                'U-':dict(home='L1-', active_operation=None)},
        trucks={'T1':dict(onboard_weight=0)}, orders={})
    candidate = dict(feasible=True, allowed=True,
        plan=dict(new_trucks=[], actions=[], decisions={}),
        state=SimpleNamespace(trucks={'T1':dict(return_time=9)}, orders={}),
        trace=[dict(kind='road_started', actor='T1', source='L1+', at=8)])
    return known, candidate


class UnifiedFeatureTests(unittest.TestCase):
    def test_groups_names_and_units_cover_seven_candidates(self):
        self.assertEqual(sorted(i for group in FEATURE_GROUPS.values() for i in group), list(range(7)))
        self.assertEqual(len(FEATURE_GROUPS), 6)
        self.assertEqual(len(FEATURE_NAMES), len(FEATURE_UNITS))
        known, candidate = fixture()
        x = features(known, candidate, 3)
        self.assertEqual(len(x), 7)
        self.assertTrue(all(isfinite(v) for v in x))

    def test_known_analytic_coverage_and_h_only_partitions_arrivals(self):
        known, candidate = fixture()
        # Releases 1..4 cover both lag samples, 5..6 cover one: total 5.
        x = features(known, candidate, 3)
        self.assertEqual(x[:3], (3., 2., 0.))
        self.assertEqual(x[4], .5)
        self.assertEqual(x[5], 0.)
        self.assertEqual(x[6], 1.)  # no D commitments
        y = features(known, candidate, 5)
        self.assertEqual(y[:2], (4.5, .5))
        self.assertEqual(x[2:], y[2:])
        # Requests in the first h minutes can use a later truck departure.
        self.assertGreater(x[0], 0)

    def test_same_city_overlapping_stations_and_trucks_do_not_duplicate_demand(self):
        known, candidate = fixture()
        before = features(known, candidate, 3)
        known.trucks['T2'] = dict(onboard_weight=0)
        candidate['state'].trucks['T2'] = dict(return_time=9)
        candidate['trace'].append(dict(kind='road_started', actor='T2', source='L1-', at=8))
        self.assertEqual(before, features(known, candidate, 3))

    def test_added_chain_is_incremental_to_existing_city_union(self):
        known, candidate = fixture()
        candidate['plan']['new_trucks'] = [dict(id='T1')]
        x = features(known, candidate, 3)
        self.assertEqual(x[:3], (0., 0., 5.))
        candidate['plan']['new_trucks'] = [dict(id='T2')]
        candidate['state'].trucks['T2'] = dict(return_time=10)
        candidate['trace'].append(dict(kind='road_started', actor='T2', source='L1-', at=9))
        x = features(known, candidate, 3)
        self.assertEqual(x[:3], (3., 2., 1.))

    def test_drone_tail_congestion_reduces_chain_opportunity(self):
        known, candidate = fixture()
        before = features(known, candidate, 3)
        candidate['trace'].insert(0, dict(kind='drone_handling_started', actor='U+', at=1, end=7))
        after = features(known, candidate, 3)
        self.assertEqual(sum(after[:2]), 0.)
        self.assertGreater(after[3], before[3])

    def test_capacity_uses_load_unload_trace_and_mean_weight_threshold(self):
        known, candidate = fixture()
        known.trucks['T1']['onboard_weight'] = 50
        self.assertEqual(features(known, candidate, 3)[4], .25)
        candidate['trace'].insert(0, dict(kind='truck_unload_done', actor='T1',
                                        quantities={'known_D':50}, at=1))
        self.assertEqual(features(known, candidate, 3)[4], .5)
        known.trucks['T1']['onboard_weight'] = 100
        candidate['trace'] = candidate['trace'][1:]
        self.assertEqual(features(known, candidate, 3)[0:3], (0., 0., 0.))

    def test_late_exposure_uses_earliest_covering_return_and_only_future_pickups(self):
        known, candidate = fixture()
        candidate['state'].trucks['T1']['return_time'] = 20
        x = features(known, candidate, 3)
        # Covered releases 1..4 at full mass, 5..6 at half mass.
        expected = (9+8+7+6+5*.5+4*.5)/10
        self.assertAlmostEqual(x[5], expected)
        known.trucks['T2'] = dict(onboard_weight=0)
        candidate['state'].trucks['T2'] = dict(return_time=9)
        candidate['trace'].append(dict(kind='road_started', actor='T2', source='L1-', at=8))
        self.assertEqual(features(known, candidate, 3)[5], 0.)
        candidate['state'].orders['not_a_known_order'] = dict(kind='P', completed_at=999999)
        self.assertEqual(features(known, candidate, 3)[5], 0.)

    def test_remaining_delivery_slack_changes_with_candidate_not_known_fines(self):
        known, candidate = fixture()
        known.orders['D'] = dict(kind='D', city='C1', deadline=10, completed_at=None)
        candidate['state'].orders['D'] = dict(completed_at=8)
        self.assertEqual(features(known, candidate, 3)[6], .2)
        candidate['state'].orders['D']['completed_at'] = 9
        self.assertEqual(features(known, candidate, 3)[6], .1)
        known.orders['D']['completed_at'] = 0
        self.assertEqual(features(known, candidate, 3)[6], 1.)

    def test_forecast_and_horizon_validation(self):
        known, candidate = fixture()
        for h in (0, -1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                features(known, candidate, h)
        known.public['revelation_closed'] = True
        self.assertEqual(features(known, {}, 3), (0.,)*7)
        known.public['revelation_closed'] = False
        known.now = 10
        self.assertEqual(features(known, {}, 3), (0.,)*7)
        known.now = 0
        del known.public['demand_model']['spatial_service_lags_by_city']
        with self.assertRaisesRegex(ValueError, 'historical spatial'):
            features(known, candidate, 3)

    def test_hidden_suffix_and_sunk_wait_history_are_not_feature_inputs(self):
        known, candidate = fixture()
        before = features(known, candidate, 3)
        class ForbiddenRead:
            def __iter__(self):
                raise AssertionError('hidden future was read')
            def __getattribute__(self, name):
                raise AssertionError('hidden future was read')
        known._hidden = ForbiddenRead()
        known._queue = ForbiddenRead()
        known.initial = ForbiddenRead()
        known.ledger = ForbiddenRead()
        known._station_wait_memory = ForbiddenRead()
        candidate['state']._hidden = ForbiddenRead()
        candidate['state'].ledger = ForbiddenRead()
        self.assertEqual(before, features(known, candidate, 3))

    def test_real_candidate_is_finite_and_inputs_unchanged(self):
        from uniform_wait_worlds import make_world
        from e1_sim.online import prepare_online, known_clone, remaining_plan, evaluate_plan
        world = make_world(609, 0, 12, .5, 'dual', history_days=2, improvement_budget=0)
        environment, _ = prepare_online(world['input'], world['configuration'], world['network'], full_day=True)
        known = known_clone(environment)
        candidate = evaluate_plan(known, remaining_plan(known))
        self.assertTrue(candidate['feasible'], candidate.get('reason'))
        public_before, plan_before = deepcopy(known.public), deepcopy(candidate['plan'])
        self.assertTrue(all(isfinite(v) for v in features(known, candidate, 5)))
        self.assertEqual(public_before, known.public)
        self.assertEqual(plan_before, candidate['plan'])


if __name__ == '__main__':
    unittest.main()
