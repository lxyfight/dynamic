import unittest
from copy import deepcopy
from profit_dispatch_timing import with_departure_delay
from profit_diagnosis import instance,specification
from run_multiscale import CFG
from e1_sim.quantities import InvalidAction

class DepartureTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp=specification('timing-test',2,0,0,4,8,i0_budget=0)
        cls.base=instance(CFG,**cls.sp)
    def test_translation_preserves_physical_orders_and_costs(self):
        original=deepcopy(self.base);shifted=with_departure_delay(self.base,240)
        self.assertEqual(self.base,original)
        self.assertEqual(shifted['input']['orders'],self.base['input']['orders'])
        self.assertEqual(shifted['input']['external_events'],self.base['input']['external_events'])
        self.assertEqual(shifted['input']['public_information'],self.base['input']['public_information'])
        self.assertTrue(shifted['initial_timing']['static_cost_unchanged'])
        delta=shifted['initial_timing']['actual']
        for a,b in zip(self.base['input']['actions'],shifted['input']['actions']):
            self.assertAlmostEqual(float(b['at'])-float(a['at']),delta)
    def test_large_delay_is_capped_by_known_delivery_slack(self):
        shifted=with_departure_delay(self.base,10000)
        self.assertLessEqual(shifted['initial_timing']['actual'],shifted['initial_timing']['minimum_delivery_slack'])
    def test_hidden_pickups_do_not_affect_initial_timing(self):
        a=with_departure_delay(self.base,240)
        b=with_departure_delay(instance(CFG,**dict(self.sp,pickup_seed=999999)),240)
        self.assertEqual(a['i0_hash'],b['i0_hash']);self.assertEqual(a['initial_timing'],b['initial_timing'])
    def test_negative_delay_rejected(self):
        with self.assertRaises(InvalidAction):with_departure_delay(self.base,-1)
    def test_decimal_time_translation_with_many_successive_sorties(self):
        world=instance(CFG,**specification('diagnosis',2,0,0,20,8))
        shifted=with_departure_delay(world,240)
        self.assertTrue(shifted['initial_timing']['static_cost_unchanged'])
        self.assertTrue(all(isinstance(a['at'],str) for a in shifted['input']['actions']))
