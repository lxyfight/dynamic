"""Station decision order and public memory for repeated active waiting."""
from copy import deepcopy
from decimal import Decimal
import json
import unittest

from e1_sim.learning import wait_pool
from e1_sim.online import evaluate_plan, install_plan, known_clone, prepare_online, remaining_plan
from e1_sim.online_examples import make_world
from e1_sim.quantities import InvalidAction
from e1_sim.runner import DEFAULT_SPEC


PACKAGE = json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
CFG = PACKAGE['configuration']
NET = PACKAGE['networks']['fixed_two_city']


class Captured(Exception):
    pass


class UniformWaitEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, _ = make_world(PACKAGE, ())
        replay, _ = prepare_online(raw, CFG, NET, full_day=True)
        replay.run(stop_at_terminal=True)
        cls.direct = replay
        cls.passed = next(e for e in replay.trace if e['kind'] == 'truck_arrived'
                          and e['destination'] != 'depot'
                          and any(r['kind'] == 'road_started' and r['actor'] == e['actor']
                                  and r['at'] == e['at'] and r['source'] == e['destination']
                                  for r in replay.trace))

    def capture(self, releases=(100,)):
        raw, _ = make_world(PACKAGE, releases)
        state, driver = prepare_online(raw, CFG, NET, full_day=True)
        def hook(env):
            if env.now == self.passed['at']:
                raise Captured()
            driver.on_event(env)
        with self.assertRaises(Captured):
            state.run(decision_hook=hook, stop_at_terminal=True)
        return state, driver

    def test_same_time_request_is_revealed_before_unplanned_departure(self):
        state, driver = self.capture((self.passed['at'], 100))
        tid = self.passed['actor']
        self.assertIn('P1', state.orders)
        self.assertEqual(state.trucks[tid]['location'], self.passed['destination'])
        self.assertIsNone(state.trucks[tid]['active_operation'])
        self.assertEqual(driver.decision_reason(state), 'request_batch')
        self.assertFalse(any(e['kind'] == 'road_started' and e['actor'] == tid
                             and e['at'] == state.now for e in state.trace))
        driver.on_event(state)
        self.assertEqual(state._driver_calls[state.now], 1)

    def test_unplanned_arrival_without_new_request_offers_wait(self):
        state, driver = self.capture()
        self.assertEqual(driver.decision_reason(state), 'station_departure_or_wait')
        known = known_clone(state)
        base = evaluate_plan(known, remaining_plan(known))
        self.assertTrue(base['feasible'])
        pool = wait_pool(known, [base], review_interval=5)
        self.assertTrue(any(c['plan'].get('wait_reviews', {}).get(self.passed['actor']) == known.now + 5
                            for c in pool))

    def test_no_hook_replay_still_passes_and_finishes(self):
        self.assertTrue(self.direct.is_terminal())
        self.assertFalse(self.direct._online_station_decisions)
        tid, at = self.passed['actor'], self.passed['at']
        self.assertTrue(any(e['kind'] == 'road_started' and e['actor'] == tid and e['at'] == at
                            for e in self.direct.trace))
        self.assertFalse(any(a['kind'] == 'truck_continue_circuit' and a.get('actor') == tid
                             for a in self.direct._actions.get(at, [])))

    def test_started_road_leg_remains_locked(self):
        state, _ = self.capture()
        tid = self.passed['actor']
        state.apply(dict(kind='truck_continue_circuit', actor=tid))
        before = deepcopy(state.trucks[tid])
        with self.assertRaisesRegex(InvalidAction, 'locked active operation'):
            state.apply(dict(kind='truck_continue_circuit', actor=tid))
        self.assertEqual(before, state.trucks[tid])

    def test_active_wait_counts_elapsed_time_and_repeated_choices(self):
        state, _ = self.capture()
        known = known_clone(state)
        tid, start = self.passed['actor'], known.now
        plan = remaining_plan(known)
        for action in plan['actions']:
            if action.get('actor') == tid:
                action['at'] += 5
        plan['wait_reviews'] = {tid: start + 5}
        install_plan(known, plan)
        known._advance(start + 2)
        memory = known._station_wait_memory[tid]
        self.assertEqual((memory['active_wait_minutes'], memory['review_count']), (Decimal(2), 1))
        install_plan(known, plan)
        self.assertEqual(known._station_wait_memory[tid]['review_count'], 1,
                         'keeping a reservation is not a new choice')
        for action in plan['actions']:
            if action.get('actor') == tid:
                action['at'] += 5
        plan['wait_reviews'] = {tid: start + 10}
        install_plan(known, plan)
        known._advance(start + 5)
        memory = known._station_wait_memory[tid]
        self.assertEqual((memory['active_wait_minutes'], memory['review_count']), (Decimal(5), 2))
        clone = known_clone(known)
        self.assertEqual(clone._station_wait_memory, known._station_wait_memory)
        clone._station_wait_memory[tid]['review_count'] += 1
        self.assertEqual(memory['review_count'], 2)
        known.apply(dict(kind='truck_continue_circuit', actor=tid))
        ended = known._station_wait_history[-1]
        self.assertEqual(ended['active_wait_minutes'], Decimal(5))
        self.assertEqual(ended['departed_at'], start + 5)
        self.assertNotIn(tid, known._wait_reviews)

    def test_known_cargo_reservation_is_not_active_wait(self):
        cases = {c['id']: c for c in PACKAGE['cases']}
        data = deepcopy(cases['E03-BM']['input'])
        data['stop_time'] = 162
        data['actions'] = [dict(at=110, kind='truck_load', actor='T1', orders=['P1', 'P2'], station='L1+'),
                           dict(at=115.5, kind='truck_continue_circuit', actor='T1')]
        data['public_information'].update(revelation_end=162)
        state, driver = prepare_online(data, CFG, NET)
        state.run(decision_hook=driver.on_event, stop_at_terminal=True)
        visit = next(v for v in state._station_wait_history if v['truck'] == 'T1' and v['station'] == 'L1+')
        self.assertEqual((visit['active_wait_minutes'], visit['review_count']), (Decimal(0), 0))


if __name__ == '__main__':
    unittest.main()
