import unittest
from copy import deepcopy
from random import Random

from e1_sim.information import pickup_deadline_lags, validate_public
from e1_sim.quantities import InvalidAction
from e1_sim.unknown_demand import public_law, sample_orders
from uniform_wait_worlds import make_world


class PickupDeadlineDistributionTests(unittest.TestCase):
    def test_per_order_draws_are_independent_of_geography_and_reproducible(self):
        for mode in ('p2', 'fixed_count'):
            options = dict(arrival_mode=mode, history_days=1, improvement_budget=0)
            first = make_world(710, 1, 30, .9, 'uniform', **options)
            spatial_pair = make_world(710, 1, 30, .9, 'dual', **options)
            expected_marks = []
            rng = Random(first['seeds']['pickup_deadlines'])
            for order in first['input']['orders']:
                if order['kind'] == 'P':
                    expected_marks.append((order['release'], order['weight'], rng.choice((240, 300, 360))))
                else:
                    self.assertEqual((order['release'], order['deadline']), (0, 660))
            self.assertEqual({lag for _, _, lag in expected_marks}, {240, 300, 360})
            for world in (first, spatial_pair):
                marks = [(o['release'], o['weight'], o['deadline'] - o['release'])
                         for o in world['input']['orders'] if o['kind'] == 'P']
                self.assertEqual(marks, expected_marks)
                law = world['input']['public_information']
                self.assertEqual(pickup_deadline_lags(law['demand_model']['attributes']), (240, 300, 360))
                self.assertNotIn('seeds', law)
            law = first['input']['public_information']
            simulated = sample_orders(law, 0, Random(11), first['network'], first['configuration'])
            self.assertEqual({o['deadline'] - o['release'] for o in simulated.values()}, {240, 300, 360})

    def test_legacy_scalar_law_and_invalid_mixture(self):
        law = public_law(1, 5)
        self.assertEqual(pickup_deadline_lags(law['demand_model']['attributes']), (360,))
        for support in ([], [240, 240], [0, 300], [False, 300], [240, '300']):
            altered = deepcopy(law)
            attrs = altered['demand_model']['attributes']
            attrs.pop('pickup_deadline_lag')
            attrs['pickup_deadline_lags'] = support
            with self.assertRaises(InvalidAction):
                validate_public(altered)


if __name__ == '__main__':
    unittest.main()
