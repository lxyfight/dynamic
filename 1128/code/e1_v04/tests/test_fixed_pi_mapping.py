import unittest,json
from copy import deepcopy
from e1_sim.runner import DEFAULT_SPEC,build_replay,execute_case,ROOT
from e1_sim.pi_adapter import fixed_feasibility
from e1_sim.quantities import plain

P=json.loads(DEFAULT_SPEC.read_text(encoding='utf8'));C={c['id']:c for c in P['cases']}

class FixedPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s=build_replay(P,C['E02']);s.run();cls.trace=plain(s.trace)
    def solve(self,trace=None,initial=None,batch='BM'):
        return fixed_feasibility(P['configuration'],P['networks']['fixed_two_city'],initial or C['E02']['input'],trace or self.trace,batch)
    def test_two_directions_original_witness(self):
        self.assertEqual(execute_case(P,C['E31'])['status'],'PASS')
    def test_loading_before_pickup_ready_rejected(self):
        t=deepcopy(self.trace)
        for e in t:
            if e['kind']=='truck_load_started':e['at']-=1;e['end']-=1
        self.assertEqual(self.solve(t)['status'],'INFEASIBLE')
    def test_wrong_handling_duration_rejected(self):
        t=deepcopy(self.trace)
        next(e for e in t if e['kind']=='truck_unload_started')['end']-=.5
        self.assertEqual(self.solve(t)['status'],'INFEASIBLE')
    def test_fractional_replenishment_rejected(self):
        t=deepcopy(self.trace)
        next(e for e in t if e['kind']=='truck_unload_started')['quantities']['D1']=9.5
        self.assertEqual(self.solve(t)['status'],'ERROR')
    def test_service_before_release_rejected(self):
        initial=deepcopy(C['E02']['input']);next(o for o in initial['orders'] if o['kind']=='P')['release']=22
        self.assertEqual(self.solve(initial=initial)['status'],'INFEASIBLE')
    def test_multibatch_full_day_BM_accepted_B1_rejected(self):
        data=json.loads((ROOT/'data/fixtures/wtq_7000_bm_15.json').read_text(encoding='utf8'))
        # This trace is physically feasible PI input even though its online WTQ
        # label failed acceptance; PI imposes no WTQ waiting restriction.
        self.assertEqual(self.solve(data['trace'],data['input'],'BM')['status'],'FEASIBLE')
        self.assertEqual(self.solve(data['trace'],data['input'],'B1')['status'],'INFEASIBLE')

if __name__=='__main__':unittest.main()
