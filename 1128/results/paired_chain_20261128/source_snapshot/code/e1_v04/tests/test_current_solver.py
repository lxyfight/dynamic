import unittest
from copy import deepcopy
from current_solver import solve
from profit_diagnosis import instance,specification
from run_multiscale import CFG
from e1_sim.quantities import InvalidAction

class CurrentEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world=instance(CFG,**specification('entry-test',2,0,0,4,0,i0_budget=0))
    def test_does_not_retime_started_truck(self):
        world=deepcopy(self.world);world['input']['trucks'][0]['active_operation']=dict(kind='road')
        with self.assertRaisesRegex(InvalidAction,'started'):solve(world,theta=0)
    def test_does_not_retime_started_drone(self):
        world=deepcopy(self.world);world['input']['drones'][0]['active_operation']=dict(kind='loading')
        with self.assertRaisesRegex(InvalidAction,'committed'):solve(world,theta=0)
