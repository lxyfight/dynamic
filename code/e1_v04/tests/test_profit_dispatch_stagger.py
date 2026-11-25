import unittest
from copy import deepcopy
from profit_dispatch_stagger import stagger
from profit_diagnosis import instance,specification
from run_multiscale import CFG
from e1_sim.quantities import dec

class StaggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp=specification('stagger-test',2,0,0,4,8,i0_budget=0)
        cls.cfg=dict(CFG,truck_capacity=60)
        cls.base=instance(cls.cfg,**cls.sp)
    def test_multiple_trips_keep_first_and_spread_later_departure(self):
        before=deepcopy(self.base);out=stagger(self.base)
        self.assertEqual(before,self.base)
        self.assertTrue(out['initial_stagger']['active'])
        offsets=list(out['initial_stagger']['offsets'].values())
        self.assertEqual(dec(offsets[0]),0)
        self.assertTrue(any(dec(x)>0 for x in offsets[1:]))
        self.assertEqual(out['input']['orders'],self.base['input']['orders'])
        self.assertEqual(out['input']['external_events'],self.base['input']['external_events'])
        self.assertEqual(len(out['input']['trucks']),len(self.base['input']['trucks']))
        self.assertTrue(out['initial_stagger']['static_cost_unchanged'])
    def test_one_trip_is_exact_noop(self):
        base=instance(CFG,**self.sp);out=stagger(base)
        self.assertFalse(out['initial_stagger']['active'])
        self.assertEqual(out['input'],base['input']);self.assertEqual(out['input_hash'],base['input_hash'])
    def test_hidden_requests_do_not_change_stagger(self):
        a=stagger(self.base);b=stagger(instance(self.cfg,**dict(self.sp,pickup_seed=12345)))
        self.assertEqual(a['initial_stagger'],b['initial_stagger']);self.assertEqual(a['i0_hash'],b['i0_hash'])
    def test_more_than_two_trips_complete_with_same_static_cost(self):
        world=instance(self.cfg,**dict(self.sp,deliveries_per_city=10));out=stagger(world)
        self.assertGreater(len(out['initial_stagger']['offsets']),2)
        self.assertTrue(out['initial_stagger']['static_cost_unchanged'])
