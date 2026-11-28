from copy import deepcopy
from decimal import Decimal
import json
import unittest

from e1_sim.initializer import digest
from e1_sim.online import (MyopicPolicy, TailManager, commit_decisions, evaluate_plan,
                           known_clone, prepare_online, remaining_plan)
from e1_sim.online_examples import make_world, run_world
from e1_sim.quantities import InvalidAction
from e1_sim.runner import DEFAULT_SPEC, POLICY_IDS, execute_case

PACKAGE = json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
CASES = {c['id']:c for c in PACKAGE['cases']}


def prepared(world, **kwargs):
    return prepare_online(world, PACKAGE['configuration'], PACKAGE['networks']['fixed_two_city'], **kwargs)


class OnlineTests(unittest.TestCase):
    def test_nine_policy_fixtures_execute_original_assertions(self):
        for cid in sorted(POLICY_IDS):
            with self.subTest(case=cid):
                r = execute_case(PACKAGE, deepcopy(CASES[cid]))
                self.assertEqual(r['status'], 'PASS', r['assertions'])

    def test_wk_accepts_and_completes_mixed_sortie_with_audited_cost(self):
        world,_ = make_world(PACKAGE)
        state,controller,report = run_world(PACKAGE,world)
        self.assertTrue(state.orders['P1']['accepted'])
        self.assertEqual(report['ledger']['net'], -184.6)
        self.assertEqual(state.absorption_time, Decimal('77'))
        self.assertEqual(controller.tail.replans_after_activation, 1)

    def test_w0_rejects_wait_dependent_pickup(self):
        world,_ = make_world(PACKAGE)
        state,_,report = run_world(PACKAGE,world,waiting='W0')
        self.assertFalse(state.orders['P1']['accepted'])
        self.assertEqual(report['ledger']['net'], -227.1)

    def test_bm_two_batches_and_b1_preserves_first_accepted_request(self):
        world,_ = make_world(PACKAGE,(1,24.6))
        bm,_,rbm = run_world(PACKAGE,world)
        b1,_,rb1 = run_world(PACKAGE,world,batch_permission='B1')
        self.assertTrue(bm.orders['P2']['accepted'])
        self.assertFalse(b1.orders['P2']['accepted'])
        self.assertIsNotNone(b1.orders['P1']['completed_at'])
        self.assertEqual(max(rbm['truck_loading_batches'].values()),2)
        self.assertEqual(max(rb1['truck_loading_batches'].values()),1)
        self.assertEqual(rbm['ledger']['net'],-156.9)
        self.assertEqual(rb1['ledger']['net'],-200.6)

    def test_new_reveal_during_unload_does_not_restart_locked_operation(self):
        world,_ = make_world(PACKAGE,(1,24.6))
        state,_,report = run_world(PACKAGE,world)
        events = [e for e in report['trace'] if e['kind']=='drone_handling_started' and not e['loading']]
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]['at'],24.5)
        self.assertEqual(state.orders['P1']['completed_at'], Decimal('83.25'))

    def test_same_clock_batch_admitted_once_with_complete_partition(self):
        world,_ = make_world(PACKAGE,(1,1))
        state,c,_ = run_world(PACKAGE,world)
        logs = [r for r in c.decisions if r['at']==1]
        self.assertEqual(len(logs),1)
        self.assertEqual(set(logs[0]['decisions']),{'P1','P2'})
        self.assertEqual(len([e for e in state.trace if e['kind'] in ('accept','reject')]),2)

    def test_last_rejection_processed_before_absorption(self):
        world,_ = make_world(PACKAGE)
        world.update(orders=[o for o in world['orders'] if o['kind']=='P'],drones=[],trucks=[],inventory=[],actions=[])
        state,c,_ = run_world(PACKAGE,world)
        self.assertFalse(state.orders['P1']['accepted'])
        self.assertEqual(state.absorption_time,1)
        self.assertTrue(c.tail.activated)
        self.assertEqual(state.ledger['rejection'],16)

    def test_unknown_count_waits_for_public_revelation_end(self):
        world,_ = make_world(PACKAGE,unknown_count=True,revelation_end=100)
        state,c,_ = run_world(PACKAGE,world)
        self.assertEqual(state.absorption_time,100)
        self.assertEqual(c.decisions[-1]['at'],100)
        self.assertEqual(c.tail.replans_after_activation,1)

    def test_public_zero_at_start_executes_delivery_to_completion(self):
        world,_ = make_world(PACKAGE,())
        state,c,_ = run_world(PACKAGE,world)
        self.assertEqual(len(c.decisions),1)
        self.assertEqual(c.decisions[0]['at'],0)
        self.assertIsNotNone(state.orders['D1']['completed_at'])
        self.assertEqual(state.absorption_time,Decimal('63.25'))

    def test_declared_end_before_reveal_is_invalid_world(self):
        world,_ = make_world(PACKAGE,unknown_count=True,revelation_end=0)
        with self.assertRaisesRegex(InvalidAction,'declared revelation end'):
            prepared(world)

    def test_duplicate_declared_end_not_scheduled_twice(self):
        world,_ = make_world(PACKAGE,unknown_count=True)
        world['external_events'].append(dict(kind='revelation_end',at=100))
        state,_ = prepared(world)
        self.assertEqual(sum(e[3]=='revelation_end' for e in state._queue),1)

    def test_policy_boundary_removes_future_request_and_event_data(self):
        world,_ = make_world(PACKAGE,(1,24.6))
        state,_ = prepared(world)
        known=known_clone(state)
        self.assertEqual(known._hidden,{})
        self.assertEqual({o['id'] for o in known.initial['orders']},{'D1'})
        self.assertEqual(known.initial['external_events'],[])
        self.assertEqual(known.stop,known.now)
        self.assertFalse(any(e[3]=='request_reveal' for e in known._queue))
        with self.assertRaisesRegex(InvalidAction,'hidden future'):
            MyopicPolicy().decide(state)

    def test_missing_admission_partition_is_atomic(self):
        world,_ = make_world(PACKAGE,())
        state,_ = prepared(world)
        before=digest(state.public_snapshot())
        with self.assertRaisesRegex(InvalidAction,'partition'):
            commit_decisions(state,{'not_revealed':True})
        self.assertEqual(digest(state.public_snapshot()),before)

    def test_candidate_oracle_cannot_mutate_environment(self):
        world,_ = make_world(PACKAGE,())
        state,_ = prepared(world)
        before=deepcopy(state.__dict__)
        known=known_clone(state)
        good=remaining_plan(known)
        self.assertTrue(evaluate_plan(known,good)['feasible'])
        bad=deepcopy(good);bad['actions']=[]
        self.assertFalse(evaluate_plan(known,bad)['feasible'])
        self.assertEqual(state.__dict__,before)

    def test_tail_rejects_forged_certificate_and_preserves_original(self):
        world,_ = make_world(PACKAGE,())
        state,_ = prepared(world);known=known_clone(state)
        original=remaining_plan(known)
        forged=deepcopy(original)
        forged.update(id='forged',actions=[],feasibility_certificate=True,finite_completion=True,permissions_valid=True)
        tail=TailManager()
        chosen=tail.select(known,original,[forged])
        self.assertEqual(chosen,original)
        self.assertEqual(tail.select(known,chosen,[forged]),chosen)
        self.assertEqual(tail.replans_after_activation,1)

    def test_tail_timeout_never_cancels_mandatory_work(self):
        world,_ = make_world(PACKAGE,())
        state,_ = prepared(world);known=known_clone(state);original=remaining_plan(known)
        tail=TailManager()
        self.assertEqual(tail.select(known,original,[],search_status='timeout'),original)
        self.assertEqual(tail.cancelled_accepted_orders,[])

    def test_online_budget_is_enforced_and_rejection_fallback_is_complete(self):
        world,_ = make_world(PACKAGE,(1,1,1,1))
        state,c=prepared(world,full_day=True)
        c.policy=MyopicPolicy(candidate_budget=1,max_joint_requests=1)
        state.run(decision_hook=c.on_event,stop_at_terminal=True)
        self.assertTrue(state.is_terminal())
        log=next(r for r in c.decisions if r['at']==1)
        self.assertLessEqual(log['candidates_tested'],1)
        self.assertTrue(log['search_truncated'])
        self.assertEqual(len(log['decisions']),4)
        self.assertTrue(all(o['accepted'] is False or o['completed_at'] is not None for o in state.orders.values()))

    def test_new_trip_after_dispatch_cutoff_cannot_rescue_late_request(self):
        world,_ = make_world(PACKAGE,(661,))
        state,c,_=run_world(PACKAGE,world)
        self.assertFalse(state.orders['P1']['accepted'])
        self.assertTrue(all(e['at']<=660 for e in state.trace if e['kind']=='truck_dispatched'))


if __name__ == '__main__':
    unittest.main()
