"""Common cadence, information boundary and intervention replay checks."""
from copy import deepcopy
import json
import unittest

from e1_sim.online import evaluate_plan, known_clone, prepare_online, remaining_plan
from e1_sim.quantities import dec, plain
from e1_sim.runner import DEFAULT_SPEC
from e1_sim.system_wait import SystemWaitRule
from e1_sim.unknown_demand import public_law
from run_multiscale import ROOT, digest, read
from run_system_wait_pilot import baseline_run, continue_branch
from system_wait_policy import SystemIntervalPolicy, paired_candidates
from uniform_wait_worlds import make_world


ZERO_MODEL = dict(columns=[], mean=[], std=[], weights=[], intercept=0.)


class CountingRule:
    def __init__(self, h=5):
        self.rule = SystemWaitRule(mode='fixed', fixed_h=h)
        self.calls = []

    def decision(self, public, now):
        self.calls.append((deepcopy(public), now))
        return self.rule.decision(public, now)


class SystemWaitingPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = make_world(609, 0, 20, .4, 'uniform',
                               history_days=4, improvement_budget=0)
        cls.model = read(ROOT/'results/frozen_six.json')
        cls.rule = SystemWaitRule(mode='fixed', fixed_h=5)
        cls.baseline, cls.snapshot, cls.completed = baseline_run(cls.world, cls.model, cls.rule)
        if not cls.snapshot:
            raise AssertionError('development world must expose a real waiting intervention')

    def test_zero_intervention_reproduces_full_trace_and_accounts(self):
        before = deepcopy(self.snapshot['environment'].__dict__)
        row, state = continue_branch(self.snapshot, self.snapshot['base']['plan'],
                                     self.model, self.rule)
        self.assertEqual(state.trace, self.completed.trace)
        self.assertEqual(state.ledger, self.completed.ledger)
        for key in ('net', 'ledger', 'accepted', 'trace_hash', 'active_wait_minutes', 'active_reviews'):
            self.assertEqual(row[key], self.baseline[key], key)
        self.assertEqual(state._station_wait_history, self.completed._station_wait_history)
        self.assertEqual(before, self.snapshot['environment'].__dict__)

    def test_cadence_computed_once_and_shared_by_distinct_trucks(self):
        known = deepcopy(self.snapshot['known'])
        tid = self.snapshot['truck']
        extra = deepcopy(known.trucks[tid])
        extra.update(id='TEST_OTHER', onboard_weight=0, active_operation=None,
                     return_time=None, departed_at=known.now, unloading_phase='skipped',
                     completed_batches=0, active_batch=None)
        extra['location'] = next(s for s in known.network['stations'] if s != extra['location'])
        known.trucks[extra['id']] = extra
        known._begin_station_visit(extra)
        plan = deepcopy(self.snapshot['base']['plan'])
        plan['actions'].append(dict(at=known.now, actor=extra['id'], station=extra['location'],
                                    kind='truck_continue_circuit'))
        base = evaluate_plan(known, plan)
        self.assertTrue(base['feasible'], base.get('reason'))
        rule = CountingRule(h=2)
        policy = SystemIntervalPolicy(ZERO_MODEL, rule, mode='scored')
        policy.enhance = False
        policy.choose(known, [base])
        self.assertEqual(len(rule.calls), 1)
        entry = policy.wait_decisions[-1]
        self.assertIn(tid, entry['departure_actors'])
        self.assertIn(extra['id'], entry['departure_actors'])
        self.assertEqual({p['truck'] for p in entry['pairs']}, set(entry['departure_actors']))
        expected_h = min(2., float(known.public['revelation_end']-known.now))
        self.assertEqual({p['h'] for p in entry['pairs']}, {expected_h})

    def test_pair_identity_separates_intervals_and_keeps_same_base(self):
        known, base = self.snapshot['known'], self.snapshot['base']
        pairs = []
        remaining = dec(known.public['revelation_end']) - known.now
        # Distinct effective intervals even if a future fixture captures a
        # decision very close to the end of public revelation.
        for h in (min(dec(2), remaining/3), min(dec(5), remaining*2/3)):
            options, _ = paired_candidates(known, base, h, truck=self.snapshot['truck'])
            self.assertEqual(len(options), 1)
            pairs.append(options[0])
        self.assertNotEqual(pairs[0]['id'], pairs[1]['id'])
        self.assertNotEqual(pairs[0]['identity']['h'], pairs[1]['identity']['h'])
        for key in ('public_prefix', 'at', 'truck', 'visit_id', 'base_plan'):
            self.assertEqual(pairs[0]['identity'][key], pairs[1]['identity'][key])
        for pair in pairs:
            self.assertEqual(pair['candidate']['plan']['decisions'], base['plan']['decisions'])

    def test_hidden_suffix_never_changes_public_clone_or_pair_generation(self):
        original = deepcopy(self.snapshot['environment'])
        if not original._hidden:
            # Exercise the information boundary even when a future snapshot
            # selector happens to choose a state after its last actual reveal.
            oid = 'TEST_UNSEEN'
            at = (original.now + dec(original.public['revelation_end']))/2
            hidden = deepcopy(next(iter(original.orders.values())))
            hidden.update(id=oid, kind='P', accepted=None, release=at, deadline=at+360)
            original._hidden[oid] = hidden
            original._schedule(at, 'request_reveal', dict(orders=[oid]), priority=1)
        changed = deepcopy(original)
        changed._hidden = {}
        changed._queue = [e for e in changed._queue if e[3] != 'request_reveal']
        a, b = known_clone(original), known_clone(changed)
        self.assertFalse(a._hidden)
        self.assertFalse(any(e[3] in ('request_reveal', 'revelation_end') for e in a._queue))
        self.assertEqual(plain(a.__dict__), plain(b.__dict__))
        ap, _ = paired_candidates(a, self.snapshot['base'], 5, truck=self.snapshot['truck'])
        bp, _ = paired_candidates(b, self.snapshot['base'], 5, truck=self.snapshot['truck'])
        self.assertEqual([(p['identity'], p['known_plan_delta']) for p in ap],
                         [(p['identity'], p['known_plan_delta']) for p in bp])
        self.assertEqual(self.rule.decision(a.public, a.now), self.rule.decision(b.public, b.now))

    def test_no_speculative_wait_preserves_disclosed_cargo_reservation(self):
        package = json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
        case = next(c for c in package['cases'] if c['id'] == 'E03-BM')
        data = deepcopy(case['input'])
        data['public_information'] = public_law(2, 4, end=162)
        data['stop_time'] = 162
        data['actions'] = [dict(at=110, kind='truck_load', actor='T1', orders=['P1', 'P2'], station='L1+'),
                           dict(at=115.5, kind='truck_continue_circuit', actor='T1', station='L1+')]
        state, _ = prepare_online(data, package['configuration'], package['networks']['fixed_two_city'])
        known = known_clone(state)
        # This local executor fixture inherits cargo already at the station;
        # full-day online histories normally provide its observed ready time.
        for lot in known.lots:
            if lot['state'] == 'available':
                known.orders[lot['order']]['available_at'] = known.now
        plan = remaining_plan(known)
        base = evaluate_plan(known, plan)
        self.assertTrue(base['feasible'])
        self.assertFalse(evaluate_plan(known, plan, 'W0')['allowed'])
        policy = SystemIntervalPolicy(ZERO_MODEL, self.rule, mode='depart', observe=False)
        policy.enhance = False
        chosen = policy.choose(known, [base])
        self.assertEqual(chosen['plan'], plan)
        self.assertFalse(chosen['plan']['wait_reviews'])
        self.assertEqual([dec(a['at']) for a in chosen['plan']['actions'] if a['kind'] == 'truck_load'], [dec(110)])


if __name__ == '__main__':
    unittest.main()
