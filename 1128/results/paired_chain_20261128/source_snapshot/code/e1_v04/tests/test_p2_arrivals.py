import unittest
from collections import Counter
from copy import deepcopy
from random import Random

from e1_sim.information import expected_arrivals, sample_release_rows, validate_public
from e1_sim.p2_arrivals import city_probabilities, count_law, sample_slots, slot_pmf
from e1_sim.quantities import InvalidAction
from e1_sim.unknown_demand import sample_orders
from run_global_demand_pilot import arrival_bands
from uniform_wait_worlds import make_world


class P2Arrivals(unittest.TestCase):
    def test_count_pmf_calibration(self):
        for expected in (0, 40, 1000):
            mass = 0
            for band in count_law(arrival_bands('peak'), expected):
                pmf = slot_pmf(band)
                self.assertAlmostEqual(sum(pmf), 1)
                mass += (band['end']-band['start']+1)*sum(n*p for n,p in enumerate(pmf))
            self.assertAlmostEqual(mass, expected)
        peak = count_law(arrival_bands('peak'), 40)[1]
        self.assertEqual(peak['trials'], 2)
        self.assertAlmostEqual(slot_pmf(peak)[2], (60/1230)**2)

    def test_simultaneous_releases_and_zero_counts(self):
        law = [dict(start=1, end=4, trials=2, probability=.5)]
        draws = [sample_slots(law, 1, Random(i)) for i in range(256)]
        self.assertTrue(any(not d for d in draws))
        self.assertTrue(any(d.count(2) == 2 for d in draws))
        self.assertTrue(any(len(d) > 3 for d in draws))
        self.assertTrue(all(2 <= t <= 4 for d in draws for t in d))

    def test_city_shares_and_sparse_history(self):
        for pattern in ('uniform', 'dual'):
            shares = city_probabilities([20., 45., 70.], pattern)
            self.assertAlmostEqual(sum(shares.values()), 1)
            self.assertAlmostEqual(shares['C1'], shares['C3'], places=6)
            self.assertGreater(shares['C2'], 0)
        world = make_world(710, 0, 6, .5, 'dual', cities=3, temporal='peak',
                           arrival_mode='p2', history_days=1, improvement_budget=0)
        law = world['input']['public_information']
        validate_public(law)
        self.assertTrue(all(law['demand_model']['spatial_service_lags_by_city'].values()))
        self.assertAlmostEqual(sum(expected_arrivals(law,c,0) for c in law['demand_model']['rates_by_city']), 3)

    def test_realized_counts_hidden_and_comparisons_paired(self):
        def world(day, pattern):
            return make_world(710, day, 8, .5, pattern, history_days=1,
                              improvement_budget=0, arrival_mode='p2')
        first, next_day, spatial = world(0, 'uniform'), world(1,'uniform'), world(0,'dual')
        self.assertEqual(first['public_hash'], next_day['public_hash'])
        self.assertEqual(first['times_weights_hash'], spatial['times_weights_hash'])
        self.assertNotEqual(first['generation']['realized_pickups'], next_day['generation']['realized_pickups'])
        for w in (first,next_day,spatial):
            self.assertEqual(w['generation']['expected_pickups'], 4)
            self.assertEqual(w['generation']['known_deliveries'], 4)
            self.assertEqual(sum(o['kind']=='D' for o in w['input']['orders']), 4)
            slots = Counter(o['release'] for o in w['input']['orders'] if o['kind']=='P')
            self.assertEqual(dict(slots), {e['at']: len(e['orders']) for e in w['input']['external_events']})
            self.assertTrue(all(n <= 2 for n in slots.values()))

    def test_rollout_sampler_uses_the_public_law(self):
        w = make_world(702, 2, 20, .5, 'dual', temporal='flat', arrival_mode='p2',
                       improvement_budget=0, history_days=1)
        law = w['input']['public_information']
        args = [w['network'], w['configuration']]
        orders = list(sample_orders(law, 123.7, Random(8), *args).values())
        released = sample_release_rows(law, 123.7, Random(8))
        self.assertTrue(orders)
        self.assertEqual([(o['release'],o['city']) for o in orders],
                         [(o['release'],o['city']) for o in released])
        for order in orders:
            self.assertGreaterEqual(order['release'],124)
            self.assertIn(order['deadline']-order['release'], (240, 300, 360))
            self.assertTrue(6 <= order['weight'] <= 19 and order['eligible_stations'])
        altered = deepcopy(law)
        altered['demand_model']['rates_by_city']['C1'][0]['rate'] *= 2
        with self.assertRaises(InvalidAction):
            validate_public(altered)
        law['revelation_closed']=True
        self.assertEqual(sample_orders(law, 0, Random(8), *args), {})


if __name__ == '__main__':
    unittest.main()
