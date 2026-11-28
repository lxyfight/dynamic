import unittest
from random import Random

from e1_sim.information import expected_arrivals
from run_global_demand_pilot import BANDS, arrival_bands, release_slots
from uniform_wait_worlds import make_world


class TemporalWorldTests(unittest.TestCase):
    def test_default_peak_preserves_original_sampling(self):
        weights = [w for lo, hi, w in BANDS for _ in range(lo, hi + 1)]
        expected = sorted(Random(321).choices(range(1, 591), weights, k=40))
        self.assertEqual(release_slots(Random(321), 40), expected)
        self.assertEqual(release_slots(Random(321), 40, 'peak'), expected)
        with self.assertRaises(ValueError):
            arrival_bands('unknown')

    def test_time_shape_changes_releases_not_known_deliveries(self):
        a = make_world(610, 0, 12, .5, 'dual', improvement_budget=0)
        b = make_world(610, 0, 12, .5, 'dual', temporal='flat', improvement_budget=0)
        self.assertEqual(a['i0_hash'], b['i0_hash'])
        self.assertEqual(a['network'], b['network'])
        def marks(w):
            return [(o['id'], o['weight'], o['xy']) for o in w['input']['orders']]
        self.assertEqual(marks(a), marks(b))
        self.assertNotEqual(a['times_weights_hash'], b['times_weights_hash'])
        self.assertNotEqual(a['id'], b['id'])
        public = b['input']['public_information']
        for city in public['demand_model']['rates_by_city']:
            self.assertAlmostEqual(expected_arrivals(public, city, 10, 20),
                                   expected_arrivals(public, city, 210, 220))
        self.assertEqual(len(b['input']['orders']), 12)

    def test_flat_history_does_not_use_realized_count(self):
        a = make_world(610, 0, 12, .5, 'uniform', temporal='flat', improvement_budget=0)
        b = make_world(610, 0, 20, .2, 'uniform', temporal='flat', improvement_budget=0)
        self.assertEqual(a['public_hash'], b['public_hash'])
        self.assertEqual(sum(o['kind'] == 'P' for o in a['input']['orders']), 6)
        self.assertEqual(sum(o['kind'] == 'P' for o in b['input']['orders']), 4)


if __name__ == '__main__':
    unittest.main()
