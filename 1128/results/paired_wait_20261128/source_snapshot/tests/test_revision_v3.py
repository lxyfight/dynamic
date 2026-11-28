import json
import pickle
import unittest
from copy import deepcopy
from e1_sim.runner import DEFAULT_SPEC
from e1_sim.quantities import dec, InvalidAction
from e1_sim.multiscale import common_i0
from e1_sim.initial_enhancement import rebuild, packing
from e1_sim.initializer import verify_plan, public_delivery_input, digest
from e1_sim.online import prepare_online, known_clone, remaining_plan, evaluate_plan, install_plan, OnlineCoordinator
from e1_sim.learning import wait_pool
from e1_sim.enhancement import enhance_pool
from e1_sim.formal_learning import FormalPolicy

PACKAGE=json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
CFG=PACKAGE['configuration'];NET=PACKAGE['networks']['fixed_two_city']
CASES={c['id']:c for c in PACKAGE['cases']}


def delivery_fixture():
    ds=[dict(id='D'+str(i),kind='D',weight=10,release=0,deadline=660,city='C1',
             xy=[20+km,.25],eligible_stations=['L1+'],accepted=True) for i,km in [(1,2),(2,3)]]
    drones=[dict(id='U1',home='L1+',location='L1+',activated=False,onboard_weight=0,takeoff_at=None,active_operation=None)]
    return ds,drones


def batch_fixture(merged=False, fractional=False):
    raw=deepcopy(CASES['E03-BM']['input'])
    raw['stop_time']=220
    raw['public_information']=dict(future_count=0,revelation_closed=True)
    raw['inventory'][0].update(state='external_pipeline',holder='external')
    second=112.3 if fractional else 112
    raw['external_events']=[dict(at=t,kind='pickup_unload_complete',orders=[o],station='L1+') for t,o in [(108,'P1'),(second,'P2')]]
    raw['actions']=([dict(at=second,kind='truck_load',actor='T1',station='L1+',orders=['P1','P2']),
                     dict(at=dec(second)+dec('5.5'),kind='truck_continue_circuit',actor='T1')] if merged else
                    [dict(at=108,kind='truck_load',actor='T1',station='L1+',orders=['P1']),
                     dict(at=113.25,kind='truck_load',actor='T1',station='L1+',orders=['P2']),
                     dict(at=118.5,kind='truck_continue_circuit',actor='T1')])
    s,_=prepare_online(raw,CFG,NET)
    return s,known_clone(s)


def idle_fixture():
    raw=deepcopy(CASES['E03-BM']['input'])
    raw.update(orders=[],inventory=[],external_events=[],stop_time=200,
               public_information=dict(future_count=1,city_pickup_totals={'C1':1},revelation_end=120,revelation_closed=False),
               actions=[dict(at=100,kind='truck_continue_circuit',actor='T1')])
    return prepare_online(raw,CFG,NET)


class InitialEnhancementTests(unittest.TestCase):
    def test_two_delivery_sorties_merge_ten_to_six_km(self):
        ds,us=delivery_fixture()
        base,_=common_i0(CFG,NET,ds,us,improve=False)
        enhanced,_=common_i0(CFG,NET,ds,us)
        problem=public_delivery_input(CFG,NET,ds,us)
        a=verify_plan(problem,base);b=verify_plan(problem,enhanced)
        self.assertEqual(len([e for e in a['trace'] if e['kind']=='flight_started']),4)
        self.assertEqual(len([e for e in b['trace'] if e['kind']=='flight_started']),3)
        self.assertAlmostEqual(float(b['ledger']['net'])-float(a['ledger']['net']),1.2)
        self.assertTrue(b['terminal'])

    def test_tight_delivery_deadlines_retain_feasible_incumbent(self):
        ds,us=delivery_fixture();base,_=common_i0(CFG,NET,ds,us,improve=False)
        for o in ds:o['deadline']=next(a['service_complete'] for a in base['assignments'] if a['order']==o['id'])
        enhanced,_=common_i0(CFG,NET,ds,us)
        verify_plan(public_delivery_input(CFG,NET,ds,us),enhanced)

    def test_zero_budget_returns_baseline_commands(self):
        ds,us=delivery_fixture();a,_=common_i0(CFG,NET,ds,us,improve=False);b,_=common_i0(CFG,NET,ds,us,improvement_budget=0)
        self.assertEqual(a['replay_input'],b['replay_input']);self.assertEqual(b['improvement']['attempted'],0)

    def test_integer_split_replenishment_is_executable(self):
        ds,us=delivery_fixture();cfg=dict(CFG,truck_capacity=15)
        allocation=packing(ds,15,True)
        self.assertEqual(allocation,[{'D1':10,'D2':5},{'D2':5}])
        p=rebuild(cfg,NET,ds,us,{'D1':'L1+','D2':'L1+'},allocation,[['D1','D2']])
        self.assertEqual(len(next(a for a in p['assignments'] if a['order']=='D2')['allocations']),2)
        self.assertTrue(verify_plan(public_delivery_input(cfg,NET,ds,us),p)['terminal'])

    def test_rejects_fractional_allocation(self):
        ds,us=delivery_fixture()
        with self.assertRaises(InvalidAction):
            rebuild(CFG,NET,ds,us,{'D1':'L1+','D2':'L1+'},[{'D1':9.5,'D2':10},{'D1':.5}],[['D1','D2']])


class OnlineEnhancementTests(unittest.TestCase):
    def test_enhancement_budget_is_attempts_not_successes(self):
        _,k=batch_fixture();base=evaluate_plan(k,remaining_plan(k))
        pool,stats=enhance_pool(k,[base],budget=1)
        self.assertLessEqual(stats['attempted'],1);self.assertIs(pool[0],base)
        rows,zero=enhance_pool(k,[base],budget=0)
        self.assertEqual(rows,[base]);self.assertEqual(zero['attempted'],0)

    def test_merge_saves_one_minute_and_point_four_yuan(self):
        _,k=batch_fixture();before=pickle.dumps(k)
        base=evaluate_plan(k,remaining_plan(k));pool,stats=enhance_pool(k,[base])
        merged=next(c for c in pool if 'merge_batches' in c['id'])
        self.assertEqual(merged['completion'],base['completion']-1)
        self.assertAlmostEqual(float(merged['score']-base['score']),.4)
        self.assertEqual(pickle.dumps(k),before);self.assertEqual(pool[0],base)
        self.assertLessEqual(stats['attempted'],16)

    def test_early_batch_uses_idle_gap(self):
        s,k=batch_fixture(merged=True)
        k.orders['P1']['available_at']=dec(100)
        # Use the original E03 known-ready inventory and 110 reservation.
        raw=deepcopy(CASES['E03-BM']['input']);raw['actions']=[dict(at=110,kind='truck_load',actor='T1',station='L1+',orders=['P1','P2']),dict(at=115.5,kind='truck_continue_circuit',actor='T1')]
        s,_=prepare_online(raw,CFG,NET);k=known_clone(s)
        base=evaluate_plan(k,remaining_plan(k));pool,_=enhance_pool(k,[base])
        split=next(c for c in pool if 'split_batch' in c['id'])
        self.assertEqual(split['completion'],dec('161.25'))
        self.assertGreater(split['score'],base['score'])

    def test_fractional_known_reservation_is_not_rounded(self):
        s,k=batch_fixture(merged=True,fractional=True)
        driver=OnlineCoordinator(FormalPolicy(group='coordination'))
        s.run(decision_hook=driver.on_event,stop_at_terminal=True)
        self.assertEqual([e['at'] for e in s.trace if e['kind']=='truck_load_started'],[dec('112.3')])
        self.assertFalse(any(r['policy_called'] for r in driver.event_checks))

    def test_scored_selected_plan_matches_sample_and_final_commit(self):
        _,k=batch_fixture();k.public.update(future_count=1,revelation_closed=False,revelation_end=200,city_pickup_totals={'C1':3})
        policy=FormalPolicy([0]*15,group='coordination')
        plan,log=policy.decide(k)
        row=policy.selection_log[-1]
        self.assertEqual(row['plan_hash'],digest(plan))
        self.assertEqual(policy.samples[-1],(row['features'],row['J']))
        self.assertIn('merge_batches',plan['id'])
        self.assertAlmostEqual(float(evaluate_plan(k,plan)['score']),row['J'])

    def test_wrong_weight_dimension_fails_explicitly(self):
        _,k=batch_fixture();policy=FormalPolicy([0]*6,group='coordination')
        with self.assertRaises(InvalidAction):policy.decide(k)

    def test_b1_does_not_adopt_second_batch(self):
        s,k=batch_fixture(merged=True);k.batch_permission='B1'
        base=evaluate_plan(k,remaining_plan(k));pool,_=enhance_pool(k,[base])
        self.assertFalse(any('split_batch' in c['id'] for c in pool))


class WaitingReviewTests(unittest.TestCase):
    def test_real_new_request_interrupts_discretionary_wait(self):
        s,_=idle_fixture();raw=deepcopy(s.initial)
        request=deepcopy(CASES['E03-BM']['input']['orders'][0])
        request.update(accepted=None,release=108,deadline=200)
        raw['orders']=[request];raw['external_events']=[dict(at=108,kind='request_reveal',orders=[request['id']])]
        s,c=prepare_online(raw,CFG,NET)
        class WaitThenMyopic(FormalPolicy):
            def choose(self,k,rows):
                return wait_pool(k,rows)[-1] if k.now<108 else super().choose(k,rows)
        c.policy=WaitThenMyopic(group='coordination')
        s.run(decision_hook=c.on_event,stop_at_terminal=True)
        self.assertEqual([r['at'] for r in c.decisions if r['policy_called']],[100,105,108])
        self.assertEqual(next(e['at'] for e in s.trace if e['kind']=='road_started'),dec(108))
        self.assertIs(s.orders[request['id']]['accepted'],False)

    def test_known_112_point3_reservation_direct_with_future_requests(self):
        s,k=batch_fixture(merged=True,fractional=True)
        s.public.update(future_count=None,revelation_closed=False,revelation_end=200)
        driver=OnlineCoordinator(FormalPolicy(group='coordination'))
        # Unknown counts are permitted by the executor; prevent this artificial
        # local example from reaching a learned departure decision.
        calls=[]
        class Captured(Exception):pass
        def hook(env):
            calls.append((env.now,driver.decision_reason(env)))
            if env.now>dec('112.3'):raise Captured()
            driver.on_event(env)
        try:s.run(decision_hook=hook)
        except Captured:pass
        self.assertIn((dec('112.3'),None),calls)
        self.assertEqual(next(e['at'] for e in s.trace if e['kind']=='truck_load_started'),dec('112.3'))

    def test_station_unload_event_can_advance_waiting_review(self):
        s,k=batch_fixture(merged=True)
        k.public.update(future_count=1,revelation_closed=False)
        k._wait_reviews={'T1':dec(105)}
        k.trace.append(dict(at=100,kind='pickup_unload_complete',orders=['P1'],station='L1+'))
        self.assertEqual(OnlineCoordinator(FormalPolicy()).decision_reason(k),'waiting_inventory_changed')
        k.trucks['T1']['active_operation']={'kind':'truck_load','end':dec(106)}
        self.assertIsNone(OnlineCoordinator(FormalPolicy()).decision_reason(k))

    def test_only_delta_wait_and_budget_zero(self):
        s,_=idle_fixture();k=known_clone(s);base=evaluate_plan(k,remaining_plan(k))
        pool=wait_pool(k,[base]);self.assertEqual(len(pool),2)
        self.assertEqual(pool[1]['plan']['wait_reviews'],{'T1':dec(105)})
        self.assertEqual(len(wait_pool(k,[base],budget=0)),1)
        with self.assertRaises(InvalidAction):wait_pool(k,[base],review_interval=0)

    def test_repeated_wait_review_terminates_at_public_end(self):
        s,c=idle_fixture()
        class AlwaysWait(FormalPolicy):
            def choose(self,k,rows):return wait_pool(k,rows)[-1]
        c.policy=AlwaysWait()
        s.run(decision_hook=c.on_event,stop_at_terminal=True)
        self.assertEqual([r['at'] for r in c.decisions if r['policy_called']],[100,105,110,115])
        self.assertTrue(c.tail.activated);self.assertTrue(s.is_terminal())
        self.assertTrue(all(v==1 for v in s._driver_calls.values()))

    def test_earlier_request_can_end_wait_before_review(self):
        s,_=idle_fixture();k=known_clone(s)
        wait=wait_pool(k,[evaluate_plan(k,remaining_plan(k))])[-1]['plan']
        install_plan(k,wait);k.now=dec(103)
        base=evaluate_plan(k,remaining_plan(k));pool,_=enhance_pool(k,[base])
        early=next(c for c in pool if 'depart_now' in c['id'])
        self.assertEqual(min(dec(a['at']) for a in early['plan']['actions']),dec(103))
        self.assertGreater(early['score'],base['score'])

    def test_revelation_boundary_shortens_last_review_interval(self):
        s,_=idle_fixture();k=known_clone(s);k.public['revelation_end']=103
        p=wait_pool(k,[evaluate_plan(k,remaining_plan(k))])[-1]
        self.assertEqual(p['plan']['wait_reviews'],{'T1':dec(103)})


if __name__=='__main__':unittest.main()
