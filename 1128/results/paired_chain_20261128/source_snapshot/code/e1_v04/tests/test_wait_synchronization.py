import unittest,json,pickle
from copy import deepcopy
from e1_sim.runner import ROOT
from e1_sim.online import prepare_online,known_clone,remaining_plan,evaluate_plan
from e1_sim.learning import wait_pool
from e1_sim.quantities import dec

class WaitSynchronization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'data/fixtures/wait_sync_example.json'
        data=json.loads(p.read_text(encoding='utf8'))
        state,driver=prepare_online(data['input'],data['configuration'],data['network'],full_day=True)
        class Stop(Exception):pass
        def hook(env):
            if driver.decision_reason(env)=='station_departure_or_wait':
                cls.known=known_clone(env);raise Stop()
            driver.on_event(env)
        try:state.run(decision_hook=hook,stop_at_terminal=True)
        except Stop:pass
    def test_legal_wait_restored_without_mutating_live_state(self):
        k=deepcopy(self.known);before=pickle.dumps(k)
        base=evaluate_plan(k,remaining_plan(k));pool=wait_pool(k,[base])
        fixed=next(r for r in pool if r['id'].endswith('_5_synchronized'))
        self.assertAlmostEqual(float(base['score']-fixed['score']),2.)
        self.assertEqual(before,pickle.dumps(k))
        self.assertTrue(all(o['completed_at'] is not None for o in fixed['state'].orders.values() if o['accepted']))
        from e1_sim.formal_features import vector
        a,b=vector(k,base,'waiting'),vector(k,fixed,'waiting')
        self.assertEqual(len(b),19)
        self.assertGreater(b[-1],a[-1])
        self.assertGreater(b[-2],a[-2])
    def test_hard_deadline_is_not_relaxed(self):
        k=deepcopy(self.known);base=evaluate_plan(k,remaining_plan(k))
        for oid,o in base['state'].orders.items():
            if o['kind']=='D' and o['completed_at']>k.now:
                k.orders[oid]['deadline']=o['completed_at']+dec('.01')
        base=evaluate_plan(k,remaining_plan(k));self.assertTrue(base['feasible'])
        self.assertFalse(any('_synchronized' in r['id'] for r in wait_pool(k,[base])))
    def test_closed_revelation_keeps_original(self):
        k=deepcopy(self.known);k.public['future_count']=0
        rows=[evaluate_plan(k,remaining_plan(k))];self.assertIs(wait_pool(k,rows),rows)
    def test_active_truck_is_not_delayed(self):
        k=deepcopy(self.known);base=evaluate_plan(k,remaining_plan(k))
        for t in k.trucks.values():t['active_operation']={'kind':'loading','end':k.now+5}
        self.assertEqual([r['id'] for r in wait_pool(k,[base])],[base['id']])
