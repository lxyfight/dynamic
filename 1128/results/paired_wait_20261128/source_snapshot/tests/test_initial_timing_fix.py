import unittest
from copy import deepcopy
from test_revision_v3 import delivery_fixture,NET,CFG as LEGACY
from e1_sim.cost_model import current_configuration
from e1_sim.initial_enhancement import rebuild
from profit_diagnosis import instance,specification
from run_multiscale import CFG

class InitialTimingFixTests(unittest.TestCase):
    def test_sequential_sorties_with_irrational_flight_times(self):
        # The pre-fix constructor rejects this feasible single-drone sequence
        # as locked: summed relative durations round differently at absolute t.
        ds,us=delivery_fixture()
        ds[0]['xy']=[26.553381,6.753678];ds[1]['xy']=[24.395242,2.83135]
        plan=rebuild(current_configuration(LEGACY),NET,ds,us,
            {'D1':'L1+','D2':'L1+'},[{'D1':10,'D2':10}],[['D1'],['D2']])
        self.assertEqual(len(plan['assignments']),2)
        self.assertTrue(all(float(a['service_complete'])<=660 for a in plan['assignments']))
    def test_sequential_loading_after_nonterminating_return_time(self):
        ds,us=delivery_fixture()
        ds[0]['xy']=[26.553381,6.753678];ds[1].update(weight=7,xy=[24.395242,2.83135])
        third=deepcopy(ds[1]);third.update(id='D3',xy=[21.921987,5.264321]);ds.append(third)
        plan=rebuild(current_configuration(LEGACY),NET,ds,us,
            {o['id']:'L1+' for o in ds},[{'D1':10,'D2':7,'D3':7}],[['D1'],['D2','D3']])
        self.assertEqual(len(plan['assignments']),3)
    def test_drone_choice_uses_current_fee_schema(self):
        ds,us=delivery_fixture();second=deepcopy(us[0]);second['id']='U2';us.append(second)
        def actors(cfg):
            p=rebuild(cfg,NET,ds,us,{'D1':'L1+','D2':'L1+'},[{'D1':10,'D2':10}],[['D1'],['D2']])
            return {a['actor'] for a in p['replay_input']['actions'] if a['kind']=='drone_sortie'}
        self.assertEqual(len(actors(LEGACY)),1)
        self.assertEqual(len(actors(current_configuration(LEGACY))),2)
    def test_multitruck_public_D_rebuild_is_not_falsely_locked(self):
        ins=instance(CFG,**specification('stagger-confirmation-v1',6,0,0,20,5,i0_budget=2))
        records=ins['i0_improvement']['records']
        self.assertEqual(len(records),2)
        self.assertTrue(all(r['feasible'] for r in records))
