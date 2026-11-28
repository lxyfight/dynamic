import unittest
from copy import deepcopy
from profit_diagnosis import instance,specification
from run_multiscale import CFG
from e1_sim.information import expected_arrivals
from e1_sim.online import prepare_online,known_clone

class ProfitDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=specification('test',2,0,0,4,8,i0_budget=0)
        cls.base=instance(CFG,**cls.spec)
    def test_delivery_prefix_and_independent_pickups(self):
        sp=dict(self.spec,deliveries_per_city=10)
        large=instance(CFG,**sp)
        a={o['id']:o for o in self.base['input']['orders']}
        b={o['id']:o for o in large['input']['orders']}
        for oid,o in a.items():self.assertEqual(o,b[oid])
    def test_arrival_transform_preserves_marks_counts_and_public_rate(self):
        a={o['id']:o for o in self.base['input']['orders'] if o['kind']=='P'}
        for timing,lo,hi in [('early',1,295),('late',296,590)]:
            ins=instance(CFG,**dict(self.spec,timing=timing))
            b={o['id']:o for o in ins['input']['orders'] if o['kind']=='P'}
            self.assertEqual(set(a),set(b))
            for oid,o in b.items():
                self.assertTrue(lo<=o['release']<=hi)
                self.assertEqual(o['deadline']-o['release'],360)
                for k in ['weight','xy','city']:self.assertEqual(a[oid][k],o[k])
            for city in ['C1','C2']:self.assertAlmostEqual(expected_arrivals(ins['input']['public_information'],city,0),8)
    def test_hidden_realization_does_not_change_known_state(self):
        a=self.base;b=instance(CFG,**dict(self.spec,pickup_seed=987))
        def public(ins):
            state,_=prepare_online(ins['input'],ins['configuration'],ins['network'],full_day=True)
            return known_clone(state).__dict__
        self.assertEqual(a['i0_hash'],b['i0_hash']);self.assertEqual(public(a),public(b))
    def test_penalty_counterfactual_does_not_change_physical_input_or_tariff(self):
        ins=instance(CFG,**dict(self.spec,rejection_factor=0.))
        self.assertEqual(ins['input'],self.base['input'])
        self.assertEqual(ins['configuration']['revenue_per_kg'],CFG['revenue_per_kg'])
        self.assertEqual(ins['configuration']['drone_sortie_cost'],25)
        self.assertEqual(CFG['rejection_per_kg'],1.6)
    def test_episode_budget_and_independent_splits(self):
        from run_workload_training import specs,LEVELS,SEEDS
        self.assertEqual(len(specs('train',17))*len(LEVELS),80)
        train={r['layout_seed'] for s in SEEDS for r in specs('train',s)}
        valid={r['layout_seed'] for r in specs('validation')}
        confirm={r['layout_seed'] for r in specs('confirmation')}
        self.assertFalse(train&valid or train&confirm or valid&confirm)
        self.assertEqual(len(specs('confirmation')),36)
        for r in specs('confirmation'):
            rho=r['expected_pickups_per_city']/(r['deliveries_per_city']+r['expected_pickups_per_city'])
            self.assertTrue(abs(rho-.2)<1e-8 or abs(rho-.4)<1e-8)
