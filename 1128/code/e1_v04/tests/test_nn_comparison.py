from copy import deepcopy
import unittest
import numpy as np
from nn_comparison import Policy,inputs,execute,fit,predict,NAMES
from recovery_opportunity import run,RecoveryPolicy
from run_multiscale import CFG
from e1_sim.unknown_demand import make_instance
from e1_sim.online import prepare_online,known_clone

class NNComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world=make_instance(CFG,cities=2,deliveries_per_city=4,expected_pickups_per_city=4,
             layout_seed=881,day_seed=661,pickup_seed=342,identity='nn-audit')
        s,d=prepare_online(cls.world['input'],CFG,cls.world['network'],full_day=True)
        captures=[]
        def hook(s):
            if not captures and any(o['kind']=='P' for o in s.orders.values()):captures.append(known_clone(s))
            return d.on_event(s)
        s.run(decision_hook=hook,stop_at_terminal=True)
        cls.known=captures[0];p=RecoveryPolicy(0)
        evaluated,_,_=p.propose(cls.known);cls.candidate=p.final_pool(cls.known,evaluated)[0]

    def test_dimensions_and_hidden_suffix_invariance(self):
        k=deepcopy(self.known);x=inputs(k,self.candidate)
        self.assertEqual(len(x),20);self.assertTrue(np.isfinite(x).all())
        k._hidden={'fake':dict(weight=100000,release=999)}
        np.testing.assert_array_equal(x,inputs(k,self.candidate))
        self.assertNotIn('future_count',k.public)

    def test_station_and_truck_cargo_are_distinct(self):
        k=deepcopy(self.known);c=deepcopy(self.candidate)
        oid=next(o for o,v in k.orders.items() if v['kind']=='P');c['plan']['decisions'][oid]=True
        lot=next(l for l in k.lots if l['order']==oid)
        lot.update(state='available',holder='L1+');x=inputs(k,c)
        lot.update(state='onboard',holder=next(iter(k.trucks)));y=inputs(k,c)
        self.assertGreater(x[NAMES.index('pickup_at_station')],y[NAMES.index('pickup_at_station')])
        self.assertGreater(y[NAMES.index('pickup_on_truck')],x[NAMES.index('pickup_on_truck')])

    def test_closed_state_masks_network_bias(self):
        k=deepcopy(self.known);k.public['revelation_closed']=True
        model=dict(kind='linear',mean=[0]*20,std=[1]*20,weights=[np.zeros((20,1)).tolist()],biases=[[999]])
        p=Policy(model);evaluated,_,_=p.propose(k);p.choose(k,evaluated)
        self.assertEqual(p.rows[-1]['w'],0)
        self.assertEqual(p.rows[-1]['w_range'],0)
        np.testing.assert_array_equal(inputs(k,self.candidate),np.zeros(20))

    def test_zero_policy_matches_existing_common_candidate_baseline(self):
        a=execute(self.world,dict(kind='zero'));b=run(self.world,0)
        self.assertAlmostEqual(a['net'],b['net']);self.assertEqual(a['accepted'],b['accepted'])
        for row in a['rows']:self.assertAlmostEqual(row['y'],a['net']-row['j'])

    def test_same_input_pool_and_model_serialization(self):
        from e1_sim.initializer import digest
        zero=Policy(dict(kind='zero'));scalar=Policy(dict(kind='scalar',theta=36))
        a,_,_=zero.propose(self.known);b,_,_=scalar.propose(self.known)
        self.assertEqual([digest(v['plan']) for v in zero.final_pool(self.known,a)],
                         [digest(v['plan']) for v in scalar.final_pool(self.known,b)])
        rows=execute(self.world,dict(kind='zero'))
        for kind in ('linear','mlp'):
            model=fit([rows],kind,17,epochs=10)
            values=predict(model,np.array([r['x'] for r in rows['rows']]))
            self.assertTrue(np.isfinite(values).all())
            import json
            np.testing.assert_array_equal(values,predict(json.loads(json.dumps(model)),np.array([r['x'] for r in rows['rows']])))

    def test_splits_have_disjoint_layouts(self):
        from run_nn_comparison import specs
        splits=[{r['layout_seed'] for r in specs('train',s)} for s in (17,29,43)]
        splits += [{r['layout_seed'] for r in specs(split)} for split in ('valid','confirm')]
        for i,a in enumerate(splits):
            for b in splits[i+1:]:self.assertFalse(a&b)
