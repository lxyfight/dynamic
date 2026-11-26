import unittest
from copy import deepcopy

from e1_sim.information import validate_public
from e1_sim.initializer import digest
from e1_sim.online import known_clone, prepare_online
from e1_sim.system_wait import SystemWaitRule, network_expected_arrivals
from uniform_wait_worlds import make_world, pilot_cases


class UniformWaitWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = [make_world(610, day, total, share, pattern, improvement_budget=0)
                      for day, total, share, pattern in
                      ((0, 12, .5, 'uniform'), (0, 12, .5, 'dual'),
                       (1, 12, .5, 'uniform'), (0, 20, .2, 'uniform'))]

    def test_pilot_design_is_32_unique_days(self):
        specs = pilot_cases()
        self.assertEqual(len(specs), 32)
        self.assertEqual(len({digest(s) for s in specs}), 32)
        self.assertEqual({s['total'] for s in specs}, {100, 150})
        self.assertEqual({s['share'] for s in specs}, {.2, .6})
        self.assertEqual(len({s['layout_id'] for s in specs}), 2)
        self.assertEqual(len({s['day_id'] for s in specs}), 2)

    def test_spatial_pair_preserves_times_weights_but_has_own_i0(self):
        a, b = self.worlds[:2]
        self.assertEqual(a['network'], b['network'])
        self.assertEqual(a['times_weights_hash'], b['times_weights_hash'])
        self.assertNotEqual(a['i0_hash'], b['i0_hash'])
        self.assertNotEqual(a['input_hash'], b['input_hash'])

    def test_days_are_separate_but_share_layout_and_historical_forecast(self):
        a, b = self.worlds[0], self.worlds[2]
        self.assertEqual(a['network'], b['network'])
        self.assertEqual(a['public_hash'], b['public_hash'])
        self.assertNotEqual(a['seeds']['releases'], b['seeds']['releases'])
        self.assertNotEqual(a['times_weights_hash'], b['times_weights_hash'])

    def test_public_law_independent_of_actual_N_and_share(self):
        a, b = self.worlds[0], self.worlds[3]
        self.assertEqual(a['public_hash'], b['public_hash'])
        for world in self.worlds:
            raw, spec = world['input'], world['spec']
            self.assertEqual(len(raw['orders']), spec['total'])
            self.assertEqual(sum(o['kind'] == 'P' for o in raw['orders']),
                             round(spec['total'] * spec['share']))
            validate_public(raw['public_information'])
            self.assertEqual(world['input_hash'], digest(raw))
            self.assertFalse({'total', 'share', 'realized_total', 'future_count',
                              'city_pickup_totals'} & set(raw['public_information']))

    def test_high_demand_forecast_and_spatial_redistribution(self):
        a, b = self.worlds[:2]
        pa, pb = (w['input']['public_information'] for w in (a, b))
        forecast = network_expected_arrivals(pa, 0)
        self.assertGreater(forecast, 40)
        self.assertLess(forecast, 70)
        rule = SystemWaitRule()
        for now in (0, 149, 151, 350, 351, 550):
            self.assertAlmostEqual(network_expected_arrivals(pa, now, now+10),
                                   network_expected_arrivals(pb, now, now+10))
            self.assertEqual(rule.interval(pa, now), rule.interval(pb, now))

    def test_historical_space_and_time_jointly_change_common_ready_h(self):
        a,b = (w['input']['public_information'] for w in self.worlds[:2])
        ready = SystemWaitRule(signal='ready',max_h=30)
        ah = [ready.interval(a,t) for t in (0,150,350)]
        bh = [ready.interval(b,t) for t in (0,150,350)]
        self.assertGreater(len(set(ah)),1)
        self.assertNotEqual(ah,bh)
        self.assertTrue(all(2<=h<30 for h in ah+bh))
        self.assertTrue(all(len(row)==128 for row in
                            a['demand_model']['spatial_service_lags_by_city'].values()))

    def test_independent_history_seed_changes_forecast_without_changing_inputs(self):
        a = self.worlds[0]
        b = make_world(610, 0, 12, .5, 'uniform', history_seed=707,
                       improvement_budget=0)
        self.assertNotEqual(a['public_hash'], b['public_hash'])
        self.assertEqual(a['i0_hash'], b['i0_hash'])
        self.assertEqual(a['times_weights_hash'], b['times_weights_hash'])
        self.assertEqual(a['input']['orders'], b['input']['orders'])

    def test_hidden_orders_cannot_change_common_interval(self):
        a, b = deepcopy(self.worlds[0]), deepcopy(self.worlds[0])
        hidden = [o for o in b['input']['orders'] if o['kind'] == 'P']
        for o in hidden:
            o.update(release=580, deadline=940, weight=19)
        b['input']['external_events'] = [dict(at=580, kind='request_reveal',
                                             orders=[o['id'] for o in hidden])]
        known = []
        for world in (a, b):
            state, _ = prepare_online(world['input'], world['configuration'],
                                      world['network'], full_day=True)
            known.append(known_clone(state))
        self.assertEqual(known[0].__dict__, known[1].__dict__)
        for mode in ('adaptive', 'daily', 'fixed'):
            for signal in ('arrivals','ready'):
                rule = SystemWaitRule(mode=mode,signal=signal,max_h=30)
                self.assertEqual(rule.interval(known[0].public, known[0].now),
                                 rule.interval(known[1].public, known[1].now))

    def test_full_size_count_and_common_plan_are_constructible(self):
        world = make_world(610, 0, 100, .6, 'uniform', improvement_budget=0)
        self.assertEqual(len(world['input']['orders']), 100)
        self.assertEqual(sum(o['kind'] == 'P' for o in world['input']['orders']), 60)
        # common_i0 physically replays and independently audits the static plan.
        self.assertTrue(world['i0_hash'])


if __name__ == '__main__':
    unittest.main()
