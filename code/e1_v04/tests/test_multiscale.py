import unittest,json,subprocess,tempfile
from pathlib import Path
from copy import deepcopy
from random import Random
from e1_sim.pi_adapter import PI_PYTHON
from e1_sim.runner import DEFAULT_SPEC
from e1_sim.multiscale import make_instance
from e1_sim.online import prepare_online,known_clone,evaluate_plan,remaining_plan
from e1_sim.formal_learning import FormalPolicy
from e1_sim.comparators import Comparator,sample_future
from e1_sim.kernel import COSTS
from e1_sim.quantities import plain

CFG=json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))['configuration']
def instance(n=10,p=.2):return make_instance(CFG,cities=2,n=n,p=p,layout_seed=60000,day_seed=60001,identity='test')
def execute(x,policy):
    s,c=prepare_online(x['input'],x['configuration'],x['network'],full_day=True);s._record_snapshots=False;c.policy=policy
    s.run(decision_hook=c.on_event,stop_at_terminal=True)
    return s,c

class MultiscaleTests(unittest.TestCase):
    def test_nested_integer_factory_and_public_counts(self):
        a=instance(10,.2);b=instance(20,.4)
        self.assertEqual(len(a['input']['orders']),20);self.assertEqual(len(b['input']['orders']),40)
        parent={o['id'][1:]:o for o in b['input']['orders']}
        for o in a['input']['orders']:
            q=parent[o['id'][1:]];self.assertEqual((o['xy'],o['weight']),(q['xy'],q['weight']))
            self.assertIs(type(o['weight']),int)
            if o['kind']=='P':self.assertEqual(q['kind'],'P')
        self.assertEqual(a['input']['public_information']['city_pickup_totals'],{'C1':2,'C2':2})

    def test_i0_does_not_use_hidden_pickups(self):
        from e1_sim.multiscale import common_i0
        a=instance();raw=deepcopy(a['input']);orders=raw['orders']
        for o in orders:
            if o['kind']=='P':o['release']=500;o['xy']=[999,999]
        _,h=common_i0(CFG,a['network'],orders,raw['drones']);self.assertEqual(a['i0_hash'],h)

    def test_oracle_isolation_and_equivalence(self):
        x=instance();s,_=prepare_online(x['input'],CFG,x['network']);s._record_snapshots=False
        k=known_clone(s);p=remaining_plan(k);before=deepcopy(k.__dict__);r=evaluate_plan(k,p)
        self.assertTrue(r['feasible']);self.assertEqual(k.__dict__,before)
        self.assertFalse(getattr(s,'_defer_global_invariants',False))
        bad=deepcopy(p);bad['actions'].append(dict(at=0,kind='truck_load',actor=next(iter(s.trucks)),orders=['missing']))
        self.assertFalse(evaluate_plan(k,bad)['feasible']);self.assertEqual(k.__dict__,before)

    def test_cached_proposals_preserve_full_trace(self):
        x=instance();cache={};a=FormalPolicy(candidate_budget=4,group='coordination');a.proposal_cache=cache
        s,c=execute(x,a);b=FormalPolicy(candidate_budget=4,group='coordination');b.proposal_cache=cache
        t,d=execute(x,b);u,e=execute(x,FormalPolicy(candidate_budget=4,group='coordination'))
        self.assertEqual(s.trace,t.trace);self.assertEqual(t.trace,u.trace);self.assertEqual(s.ledger,u.ledger)

    def test_conditional_sampler_never_reads_hidden(self):
        x=instance();s,_=prepare_online(x['input'],CFG,x['network']);k=known_clone(s)
        a=sample_future(k,Random(7));s._hidden={};b=sample_future(known_clone(s),Random(7))
        self.assertEqual(a,b);self.assertEqual(len(a),4);self.assertTrue(all(0<o['release']<=590 for o in a.values()))

    def test_comparators_complete_real_obligations(self):
        for method in ['Greedy','RH','Rollout']:
            with self.subTest(method=method):
                s,_=execute(instance(),Comparator(method,candidate_budget=4,rollout_seconds=.05))
                self.assertTrue(s.is_terminal())

    def test_complete_rollout_round_on_tiny_case(self):
        x=make_instance(CFG,cities=2,n=10,p=.4,layout_seed=60000,day_seed=60001,identity='tiny',small_total=5)
        policy=Comparator('Rollout',candidate_budget=1,rollout_h=1,rollout_seconds=10)
        s,_=execute(x,policy);self.assertTrue(s.is_terminal())
        self.assertTrue(any(r['complete_common_rounds']==1 for r in policy.rollout_log))

    def test_pi_relaxation_upper_bound_exceeds_executable_profit(self):
        x=make_instance(CFG,cities=2,n=10,p=.4,layout_seed=60000,day_seed=60001,identity='tiny',small_total=10)
        s,_=execute(x,FormalPolicy(candidate_budget=4));profit=float(s.ledger['revenue']-sum(s.ledger[k] for k in COSTS))
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'input.json';b=Path(tmp)/'bound.json';a.write_text(json.dumps(x),encoding='utf8')
            subprocess.run([str(PI_PYTHON),'-m','e1_sim.pi_route_bound',str(a),str(b),'10'],check=True,capture_output=True)
            r=json.loads(b.read_text());self.assertTrue(r['enumeration_complete']);self.assertGreaterEqual(r['upper_bound']+1e-6,profit)

if __name__=='__main__':unittest.main()
