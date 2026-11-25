import unittest,json,pickle,math
from copy import deepcopy
from random import Random
from pathlib import Path
from e1_sim.runner import DEFAULT_SPEC
from e1_sim.unknown_demand import make_instance,public_law,sample_orders
from e1_sim.information import expected_arrivals,sample_release_rows,validate_public,closed
from e1_sim.online import prepare_online,known_clone,evaluate_plan,remaining_plan,install_plan
from e1_sim.formal_learning import FormalPolicy
from e1_sim.formal_features import vector,names
from e1_sim.resource_features import resource_windows
from e1_sim.learning import wait_pool
from e1_sim.comparators import sample_future,Comparator
from e1_sim.forecast import sample_releases
from e1_sim.initializer import digest
from e1_sim.quantities import InvalidAction,dec
from test_revision_v3 import idle_fixture,batch_fixture

CFG=json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))['configuration']
def instance(pickup_seed=5,expected=3):
    return make_instance(CFG,cities=2,deliveries_per_city=4,expected_pickups_per_city=expected,
                         layout_seed=722,day_seed=901,identity='unknown',pickup_seed=pickup_seed)
def known(x):
    s,d=prepare_online(x['input'],CFG,x['network'],full_day=True)
    return s,d,known_clone(s)

class UnknownInformationTests(unittest.TestCase):
    def test_realized_counts_not_public(self):
        x=instance();p=x['input']['public_information'];validate_public(p)
        self.assertNotIn('future_count',p);self.assertNotIn('city_pickup_totals',p)
        for key in ['future_count','city_pickup_totals','realized_total']:
            bad=deepcopy(p);bad[key]=0
            with self.assertRaises(InvalidAction):validate_public(bad)
    def test_initialization_rejects_count_smuggling(self):
        x=instance();x['input']['public_information']['future_count']=0
        with self.assertRaises(InvalidAction):known(x)
    def test_poisson_counts_variable_mean_and_no_cap(self):
        p=public_law(1,4,100);counts=[len(sample_release_rows(p,0,Random(s))) for s in range(1500)]
        self.assertGreater(len(set(counts)),5);self.assertGreater(max(counts),4)
        self.assertLess(abs(sum(counts)/len(counts)-4),.2)
    def test_independent_D_and_i0_under_hidden_change(self):
        a,b=instance(5),instance(98)
        self.assertEqual(a['i0_hash'],b['i0_hash'])
        self.assertEqual([o for o in a['input']['orders'] if o['kind']=='D'],[o for o in b['input']['orders'] if o['kind']=='D'])
        self.assertNotEqual([o for o in a['input']['orders'] if o['kind']=='P'],[o for o in b['input']['orders'] if o['kind']=='P'])
    def test_known_state_identical_with_different_hidden_streams(self):
        a=known(instance(5))[2];b=known(instance(98))[2]
        self.assertEqual(a.__dict__,b.__dict__)
    def test_candidate_features_and_action_nonanticipatory(self):
        a,b=known(instance(5))[2],known(instance(98))[2]
        # Nonzero synthetic weights deliberately test sensitivity, not only Myopic.
        w=[(-1.)**j*(j+1) for j in range(16)]
        policies=[FormalPolicy(w,group='resource_demand',candidate_budget=4,seed=3,explore=.2) for _ in range(2)]
        pa,la=policies[0].decide(a);pb,lb=policies[1].decide(b)
        self.assertEqual(digest(pa),digest(pb));self.assertEqual(policies[0].selection_log,policies[1].selection_log)
    def test_historical_law_not_actual_count_drives_forecast(self):
        a=known(instance(5))[2];b=known(instance(98))[2]
        self.assertEqual(sample_future(a,Random(101)),sample_future(b,Random(101)))
        self.assertEqual(sample_releases(0,a.public,111),sample_releases(0,b.public,111))
    def test_expected_remaining_integrates_future_intensity(self):
        p=public_law(1,10,100)
        self.assertAlmostEqual(expected_arrivals(p,'C1',40),6)
        self.assertAlmostEqual(expected_arrivals(p,'C1',40.5,45),.5)
        self.assertEqual(expected_arrivals(p,'C1',100),0)
    def test_piecewise_rate_and_integer_boundary(self):
        p=public_law(1,0,100);p['demand_model']['rates_by_city']['C1']=[dict(start=1,end=40,rate=.1),dict(start=41,end=100,rate=.2)]
        self.assertAlmostEqual(expected_arrivals(p,'C1',40),12)
        rows=sample_release_rows(p,40.7,Random(4));self.assertTrue(all(41<=r['release']<=100 for r in rows))
    def test_invalid_rate_segments_rejected(self):
        for rate in [-1,float('nan'),float('inf')]:
            p=public_law(1,0);p['demand_model']['rates_by_city']['C1'][0]['rate']=rate
            with self.assertRaises(InvalidAction):validate_public(p)
    def test_no_sampling_after_public_close(self):
        p=public_law(1,3);p['revelation_closed']=True
        self.assertEqual(sample_release_rows(p,20,Random(1)),[]);self.assertEqual(expected_arrivals(p,'C1',20),0)
    def test_mark_integer_and_draw_independent(self):
        x=instance();s,d,k=known(x);draw=sample_future(k,Random(52))
        self.assertTrue(draw)
        for o in draw.values():
            self.assertIs(type(o['weight']),int);self.assertTrue(0<o['release']<=590);self.assertTrue(o['eligible_stations'])
        self.assertFalse(set(draw)&set(s._hidden))
    def test_no_early_termination_on_empty_hidden_or_zero_forecast(self):
        s,d,k=known(instance(expected=0));self.assertFalse(s._hidden);self.assertFalse(s.is_terminal())
        d.policy=FormalPolicy(candidate_budget=4);s.run(decision_hook=d.on_event,stop_at_terminal=True)
        self.assertTrue(s.is_terminal());self.assertEqual(float(s.absorption_time),590)
        self.assertTrue(d.tail.activated)
    def test_unknown_count_wait_candidates_remain_enabled(self):
        s,d=idle_fixture();s.public=public_law(2,4,120);k=known_clone(s)
        base=evaluate_plan(k,remaining_plan(k));pool=wait_pool(k,[base])
        self.assertTrue(any('wait' in c['id'] for c in pool));self.assertFalse(d.tail.select(k,base['plan']) is None);self.assertFalse(d.tail.activated)
    def test_wait_timing_uses_horizon_not_sampled_remaining(self):
        s,d=idle_fixture();s.public=public_law(2,0,103);k=known_clone(s)
        base=evaluate_plan(k,remaining_plan(k));pool=wait_pool(k,[base])
        waits=[c for c in pool if 'wait' in c['id']];self.assertTrue(waits)
        self.assertEqual(float(waits[0]['plan']['wait_reviews']['T1']),103)
    def test_known_only_J_still_uses_finite_projection(self):
        s,d,k=known(instance());r=evaluate_plan(k,remaining_plan(k))
        self.assertTrue(r['feasible']);self.assertTrue(r['state'].is_terminal())
        self.assertFalse(closed(k.public));self.assertEqual(k.public,s.public)
    def test_new_feature_dimensions_and_old_weights_rejected(self):
        s,d,k=known(instance());r=evaluate_plan(k,remaining_plan(k))
        for g,n in [('resource',12),('resource_demand',16)]:
            x=vector(k,r,g);self.assertEqual(len(x),n);self.assertTrue(all(math.isfinite(v) for v in x))
        with self.assertRaises(ValueError):vector(k,r,'waiting')
        with self.assertRaises(InvalidAction):FormalPolicy([0.]*19,group='resource').choose(k,[r])
    def test_zero_forecast_does_not_zero_resource_features(self):
        s,d,k=known(instance(expected=0));r=evaluate_plan(k,remaining_plan(k))
        self.assertEqual(vector(k,r,'resource')[0],1)
        k.public['revelation_closed']=True;self.assertEqual(vector(k,r,'resource'),[0.]*12)
    def test_idle_window_and_capacity_units(self):
        s,d=idle_fixture();s.public=public_law(2,4,120);k=known_clone(s)
        r=evaluate_plan(k,remaining_plan(k));p=wait_pool(k,[r]);w=next(x for x in p if 'wait' in x['id'])
        a,aa=resource_windows(k,r);b,bb=resource_windows(k,w)
        self.assertAlmostEqual(sum(y-x for st,x,y,q in b)-sum(y-x for st,x,y,q in a),5)
        self.assertTrue(all(0<=q<=1 for st,x,y,q in b))
        self.assertNotEqual(vector(k,r,'resource'),vector(k,w,'resource'))
    def test_same_revealed_prefix_remains_nonanticipatory(self):
        a=instance();b=deepcopy(a);future=[o for o in b['input']['orders'] if o['kind']=='P']
        for o in future:o['release']=500;o['deadline']=860;o['weight']=6
        b['input']['external_events']=[dict(at=500,kind='request_reveal',orders=[o['id'] for o in future])]
        sa,da,ka=known(a);sb,db,kb=known(b)
        # Change an unrevealed mark/time without changing any observed input.
        self.assertEqual(ka.__dict__,kb.__dict__)
    def test_nonanticipation_after_shared_revealed_order_and_operations(self):
        a=instance();b=deepcopy(a)
        shared=deepcopy(next(o for o in a['input']['orders'] if o['kind']=='P'))
        shared.update(id='OBSERVED_P',release=10,deadline=370)
        snapshots=[];logs=[]
        for index,x in enumerate([a,b]):
            ds=[o for o in x['input']['orders'] if o['kind']=='D']
            future=deepcopy(shared);future.update(id='HIDDEN_P',release=200+index*100,deadline=560+index*100,weight=6+index)
            x['input']['orders']=ds+[deepcopy(shared)]+([future] if index else [])
            x['input']['external_events']=[dict(at=10,kind='request_reveal',orders=['OBSERVED_P'])]+([dict(at=future['release'],kind='request_reveal',orders=['HIDDEN_P'])] if index else [])
            s,d=prepare_online(x['input'],CFG,x['network'],full_day=True)
            s.stop=dec(30);s.full_day=False;d.extend_horizon=False
            d.policy=FormalPolicy(candidate_budget=4,group='resource_demand',weights=[.1]*16,seed=1)
            s.run(decision_hook=d.on_event)
            self.assertIn('OBSERVED_P',s.orders);self.assertEqual(float(s.now),30)
            snapshots.append(known_clone(s));logs.append(d.policy.selection_log)
        self.assertEqual(snapshots[0].__dict__,snapshots[1].__dict__)
        self.assertEqual(logs[0],logs[1])
        policies=[FormalPolicy(weights=[.1]*16,group='resource_demand',candidate_budget=4,seed=9,explore=.5) for _ in range(2)]
        plans=[p.decide(k)[0] for p,k in zip(policies,snapshots)]
        self.assertEqual(digest(plans[0]),digest(plans[1]))
    def test_rollout_uses_synthetic_world_not_actual_remaining(self):
        a=known(instance())[2];r=evaluate_plan(a,remaining_plan(a))
        policy=Comparator('Rollout',candidate_budget=1,enhancement_budget=0,wait_budget=0,rollout_h=1,rollout_seconds=15)
        chosen=policy.choose(a,[r]);self.assertTrue(chosen['feasible'])
        self.assertEqual(policy.rollout_log[-1]['complete_common_rounds'],1)

if __name__=='__main__':unittest.main()
