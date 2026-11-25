from copy import deepcopy
from decimal import Decimal
import json
import unittest

from e1_sim.constraints import continuation
from e1_sim.kernel import Replay
from e1_sim.online import MyopicPolicy, prepare_online, known_clone, remaining_plan, pickup_variants, evaluate_plan
from e1_sim.online_examples import make_world
from e1_sim.runner import DEFAULT_SPEC, execute_case
from e1_sim.quantities import InvalidAction

PACKAGE = json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
CASES = {c['id']: c for c in PACKAGE['cases']}
CFG, NET = PACKAGE['configuration'], PACKAGE['networks']['fixed_two_city']


class CommitmentTests(unittest.TestCase):
    def test_fixed_airborne_sequence_and_station(self):
        data = deepcopy(CASES['E11-exact']['input'])
        self.assertTrue(continuation(data, CFG)['feasible'])
        for visits, station in [(['P1','D1'], 'L1+'), (['D1'], 'L1+'), (['D1','P1'], 'L1-')]:
            with self.subTest(visits=visits, station=station):
                candidate = deepcopy(data)
                candidate['candidates'][0].update(visit_order=visits, return_station=station)
                self.assertFalse(continuation(candidate, CFG)['feasible'])

    def test_request_released_after_takeoff_cannot_be_in_commitment(self):
        data = deepcopy(CASES['E11-exact']['input'])
        data['orders'][1]['release'] = 110
        self.assertFalse(continuation(data, CFG)['feasible'])

    def test_unflown_loaded_sortie_can_pickup_before_delivery_when_capacity_allows(self):
        data = deepcopy(CASES['E30']['input'])
        data.update(candidates=[], actions=[])
        data['orders'][1]['weight'] = 6
        state = Replay(CFG, NET, data)
        self.assertTrue(state.probe(dict(operation='drone_sortie', actor='U1', visit_order=['P1','D1']))['accepted'])
        self.assertFalse(state.probe(dict(operation='drone_sortie', actor='U1', visit_order=['P1']))['accepted'])

    def test_takeoff_queue_cannot_be_interrupted_by_new_sortie(self):
        data = deepcopy(CASES['E30']['input']); data.update(candidates=[], actions=[])
        state = Replay(CFG, NET, data)
        state.apply(dict(kind='drone_sortie', actor='U1', visit_order=['D1','P1']))
        before = deepcopy(state.__dict__)
        with self.assertRaises(InvalidAction):
            state.apply(dict(kind='drone_sortie', actor='U1', visit_order=['D1']))
        self.assertEqual(state.__dict__, before)
        self.assertEqual(state.drones['U1']['sortie_commitment']['visit_order'], ['D1','P1'])

    def test_generator_adjusts_unflown_mixed_sortie(self):
        world, _ = make_world(PACKAGE, (1,1))
        state, controller = prepare_online(world, CFG, NET)
        captured = []
        def hook(s):
            if s.now == 1:
                known = known_clone(s)
                base = remaining_plan(known); base['decisions']={'P1':True,'P2':True}
                first = next(v for v in pickup_variants(known, base, 'P1')
                             if any(a['kind']=='drone_sortie' and a['visit_order']==['D1','P1'] for a in v['actions']))
                for variant in pickup_variants(known, first, 'P2'):
                    if any(a['kind']=='drone_sortie' and set(a['visit_order'])=={'D1','P1','P2'} for a in variant['actions']):
                        if evaluate_plan(known, variant)['feasible']: captured.append(variant)
            controller.on_event(s)
        state.run(decision_hook=hook)
        self.assertTrue(captured)


class TriggerTests(unittest.TestCase):
    def test_known_110_batch_executes_without_learning_call(self):
        data = deepcopy(CASES['E03-BM']['input'])
        data['stop_time'] = 162
        data['actions'] = [dict(at=110,kind='truck_load',actor='T1',orders=['P1','P2'],station='L1+'),
                           dict(at=115.5,kind='truck_continue_circuit',actor='T1')]
        data['public_information'].update(revelation_end=162)
        state,c = prepare_online(data,CFG,NET)
        state.run(decision_hook=c.on_event, stop_at_terminal=True)
        self.assertEqual(state.orders['P1']['completed_at'], Decimal('161.5'))
        self.assertEqual(state.orders['P2']['completed_at'], Decimal('161.5'))
        self.assertFalse(any(r['at']==110 and r['policy_called'] for r in c.event_checks))
        self.assertEqual([e['at'] for e in state.trace if e['kind']=='truck_load_started'],[Decimal(110)])
        self.assertEqual([r['at'] for r in c.decisions if r['policy_called']], [115.5])

    def test_observation_events_change_neither_actions_nor_policy_calls(self):
        raw,_ = make_world(PACKAGE,(1,24.6))
        def run(data):
            s,c=prepare_online(data,CFG,NET,full_day=True)
            s.run(decision_hook=c.on_event,stop_at_terminal=True)
            return s,c
        a,ca=run(raw)
        modified=deepcopy(raw)
        modified['external_events'] += [dict(at=t,kind='observation_only') for t in [2,3,10,24.7,31,40]]
        b,cb=run(modified)
        self.assertEqual(a.trace,b.trace)
        self.assertEqual(a.ledger,b.ledger)
        self.assertEqual(ca.decisions,cb.decisions)
        self.assertEqual(a._driver_calls,b._driver_calls)

    def test_flights_load_reservations_and_tail_are_not_policy_calls(self):
        raw,_=make_world(PACKAGE,(1,24.6))
        s,c=prepare_online(raw,CFG,NET,full_day=True)
        s.run(decision_hook=c.on_event,stop_at_terminal=True)
        self.assertEqual(set(s._driver_calls),{Decimal(1),Decimal('24.6')})
        self.assertTrue(all(n==1 for n in s._driver_calls.values()))
        self.assertLess(len(s._driver_calls),len(s.trace))

    def test_initial_zero_future_enters_tail_without_policy(self):
        raw,_=make_world(PACKAGE,())
        s,c=prepare_online(raw,CFG,NET,full_day=True)
        s.run(decision_hook=c.on_event,stop_at_terminal=True)
        self.assertFalse(s._driver_calls)
        self.assertTrue(c.tail.activated)
        self.assertTrue(s.is_terminal())


if __name__=='__main__': unittest.main()
