"""Physical regression cases for local delay propagation and uniform reviews."""
from copy import deepcopy
import json
import pickle
import unittest

from e1_sim.cost_model import current_configuration
from e1_sim.kernel import Replay
from e1_sim.learning import wait_pool
from e1_sim.online import evaluate_plan, pickup_variants
from e1_sim.quantities import dec, InvalidAction
from e1_sim.retiming import retime_plan
from e1_sim.runner import DEFAULT_SPEC


def dependency_case():
    cfg = current_configuration(json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))['configuration'])
    net = dict(stations={'A': [0, 0], 'B': [3, 0]}, arcs=[
        {'from': 'depot', 'to': 'A', 'km': 2, 'min': 2, 'tolled': True},
        {'from': 'A', 'to': 'B', 'km': 2, 'min': 2, 'tolled': True},
        {'from': 'B', 'to': 'depot', 'km': 2, 'min': 2, 'tolled': True}])
    orders = [dict(id=oid, kind='D', weight=10, release=0, deadline=660,
                   accepted=True, city='C1', xy=xy, eligible_stations=[station])
              for oid, station, xy in [('D1', 'A', [0, 1]), ('D2', 'B', [3, 1])]]
    trucks = [dict(id='T1', location='A', onboard_weight=10, departed_at=0,
                   unloading_phase='completed', completed_batches=0, active_operation=None)]
    drones = [dict(id=uid, home=home, location=home, activated=False,
                   onboard_weight=0, takeoff_at=None, active_operation=None)
              for uid, home in [('U1', 'A'), ('U2', 'B')]]
    raw = dict(start_time=0, stop_time=100, orders=orders, trucks=trucks, drones=drones,
               inventory=[dict(order='D1', lot='D1', weight=10, state='available', holder='A'),
                          dict(order='D2', lot='D2', weight=10, state='onboard', holder='T1')],
               actions=[], external_events=[], public_information=dict(future_count=1,
                   revelation_end=50, revelation_closed=False))
    known = Replay(cfg, net, raw)
    plan = dict(id='base', decisions={}, new_trucks=[], wait_reviews={}, actions=[
        dict(at=0, kind='truck_continue_circuit', actor='T1', station='A'),
        dict(at=2, kind='truck_unload', actor='T1', station='B', quantities={'D2': 10}),
        dict(at=dec('2.25'), kind='truck_continue_circuit', actor='T1', station='B'),
        dict(at=0, kind='drone_load', actor='U1', orders=['D1']),
        dict(at=dec('1.25'), kind='drone_sortie', actor='U1', visit_order=['D1']),
        dict(at=dec('2.25'), kind='drone_load', actor='U2', orders=['D2']),
        dict(at=dec('3.5'), kind='drone_sortie', actor='U2', visit_order=['D2'])])
    return known, plan


class DependencyRetiming(unittest.TestCase):
    def test_wait_propagates_supply_dependency_but_not_independent_drone(self):
        known, plan = dependency_case()
        base = evaluate_plan(known, plan)
        self.assertTrue(base['feasible'])
        known.orders['D1']['deadline'] = base['state'].orders['D1']['completed_at'] + dec('.01')
        original = pickle.dumps(known)
        diagnostics = []
        pool = wait_pool(known, [evaluate_plan(known, plan)], review_interval=5,
                         departure_only=True, diagnostics=diagnostics)
        self.assertEqual(len(pool), 2, diagnostics)
        delayed = pool[1]
        self.assertTrue(delayed['state'].is_terminal())
        by_actor = lambda p, actor: [dec(a['at']) for a in p['actions'] if a['actor'] == actor]
        self.assertEqual(by_actor(delayed['plan'], 'U1'), by_actor(plan, 'U1'))
        self.assertEqual(by_actor(delayed['plan'], 'T1'), [dec(5), dec(7), dec('7.25')])
        self.assertEqual(by_actor(delayed['plan'], 'U2'), [dec('7.25'), dec('8.5')])
        self.assertEqual(original, pickle.dumps(known))
        self.assertEqual(delayed['wait_info']['base_id'], 'base')
        self.assertEqual(diagnostics[0]['action_kind'], 'truck_continue_circuit')

    def test_existing_slack_absorbs_delay(self):
        known, plan = dependency_case()
        plan['actions'][5]['at'] = 30
        plan['actions'][6]['at'] = dec('31.25')
        delayed = deepcopy(plan)
        delayed['actions'][0]['at'] = 5
        fixed = retime_plan(known, delayed, order_reference=plan)
        self.assertEqual(fixed['actions'][5]['at'], 30)
        self.assertEqual(fixed['actions'][6]['at'], dec('31.25'))
        self.assertTrue(evaluate_plan(known, fixed)['feasible'])

    def test_active_operation_and_event_queue_are_not_retimed(self):
        known, plan = dependency_case()
        known.apply(plan['actions'][3])
        del plan['actions'][3]
        before = pickle.dumps(known)
        base = evaluate_plan(known, plan)
        self.assertTrue(base['feasible'])
        pool = wait_pool(known, [base], departure_only=True)
        self.assertEqual(len(pool), 2)
        events = [e for e in pool[1]['trace'] if e['kind'] == 'drone_handling_done'
                  and e['actor'] == 'U1']
        self.assertEqual(events[0]['at'], dec('1.25'))
        self.assertEqual(before, pickle.dumps(known))

    def test_hard_deadline_rejection_is_reported_without_global_fallback(self):
        known, plan = dependency_case()
        base = evaluate_plan(known, plan)
        known.orders['D2']['deadline'] = base['state'].orders['D2']['completed_at'] + dec('.01')
        diagnostics = []
        pool = wait_pool(known, [evaluate_plan(known, plan)], departure_only=True,
                         diagnostics=diagnostics)
        self.assertEqual(len(pool), 1)
        self.assertEqual(len(diagnostics), 1)
        self.assertFalse(diagnostics[0]['feasible'])
        self.assertIn('deadline', diagnostics[0]['reason'])

    def test_uniform_callback_runs_once_and_legacy_callback_remains_supported(self):
        known, plan = dependency_case()
        calls = []
        def interval(state):
            calls.append(state.now)
            return 2
        base = evaluate_plan(known, plan)
        pool = wait_pool(known, [base], review_interval=interval, departure_only=True)
        self.assertEqual(calls, [known.now])
        self.assertEqual(pool[1]['wait_info']['h'], 2)
        legacy = wait_pool(known, [base], review_interval=lambda state, station: 2)
        self.assertEqual(legacy[1]['wait_info']['h'], 2)

    def test_departure_only_does_not_delay_current_loading(self):
        known, plan = dependency_case()
        known.orders['P1'] = dict(id='P1', kind='P', weight=6, release=dec(0), deadline=dec(100),
            accepted=True, city='C1', xy=[0, 1], eligible_stations=['A'], completed_at=None,
            available_at=dec(0), collected_at=dec(0), station='A', assigned_drone=None)
        known._new_lot('P1', 6, 'available', 'A')
        plan['actions'].insert(0, dict(at=0, kind='truck_load', actor='T1', station='A', orders=['P1']))
        plan['actions'][1]['at'] = dec('.15')
        plan = retime_plan(known, plan)
        base = evaluate_plan(known, plan)
        self.assertTrue(base['feasible'], base.get('reason'))
        self.assertEqual(len(wait_pool(known, [base], departure_only=True)), 1)
        self.assertEqual(len(wait_pool(known, [base])), 2)

    def test_past_commands_are_rejected(self):
        known, plan = dependency_case()
        known.now = dec(1)
        with self.assertRaisesRegex(InvalidAction, 'started/past'):
            retime_plan(known, plan)

    def test_pickup_insertion_keeps_independent_deadline(self):
        known, plan = dependency_case()
        # Both deliveries are already stocked. Only U1 may serve the new pickup;
        # U2's later, tightly timed delivery has no cargo/device dependency on it.
        for lot in known.lots:
            if lot['order'] == 'D2':
                lot.update(state='available', holder='B')
        known.orders['D2']['station'] = 'B'
        known.trucks['T1']['onboard_weight'] = 0
        plan['actions'] = [a for a in plan['actions'] if a['actor'] != 'T1']
        plan['actions'] += [dict(at=0, kind='truck_continue_circuit', actor='T1', station='A')]
        base = evaluate_plan(known, plan)
        self.assertTrue(base['feasible'])
        known.orders['D2']['deadline'] = base['state'].orders['D2']['completed_at'] + dec('.01')
        known.orders['P1'] = dict(id='P1', kind='P', weight=6, release=dec(0), deadline=dec(100),
            accepted=True, city='C1', xy=[0, 2], eligible_stations=['A'], completed_at=None,
            available_at=None, collected_at=None, station=None, assigned_drone=None)
        known._new_lot('P1', 6, 'customer', 'P1')
        candidates = [evaluate_plan(known, p) for p in pickup_variants(known, plan, 'P1')
                      if any(a['kind'] == 'drone_sortie' and set(a['visit_order']) == {'D1', 'P1'}
                             for a in p['actions'])]
        feasible = [r for r in candidates if r['feasible']]
        self.assertTrue(feasible, [r.get('reason') for r in candidates])
        for row in feasible:
            self.assertEqual(row['state'].orders['D2']['completed_at'],
                             base['state'].orders['D2']['completed_at'])


if __name__ == '__main__':
    unittest.main()
