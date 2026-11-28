from copy import deepcopy
from decimal import Decimal
import unittest

from e1_sim.audit import audit_ledger
from e1_sim.initializer import digest
from e1_sim.kernel import COSTS
from e1_sim.online import known_clone, prepare_online
from e1_sim.quantities import dec, plain
from paired_wait_suffix import resample_suffix
from uniform_wait_worlds import make_world


class StopAtSnapshot(Exception):
    pass


class PairedSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        world = make_world(112881, 0, 20, .6, 'uniform', arrival_mode='p2',
                           history_days=4, improvement_budget=0)
        state, controller = prepare_online(world['input'], world['configuration'],
                                            world['network'], full_day=True)
        controller.policy.candidate_budget = 4
        def hook(environment):
            if environment.now >= 20:
                cls.snapshot = deepcopy(environment)
                cls.controller = deepcopy(controller)
                raise StopAtSnapshot()
            controller.on_event(environment)
        try:
            state.run(decision_hook=hook, stop_at_terminal=True)
        except StopAtSnapshot:
            pass
        assert hasattr(cls, 'snapshot')

    def test_reproducible_suffix_independent_of_original_hidden_future(self):
        first, meta = resample_suffix(self.snapshot, 123)
        second, again = resample_suffix(self.snapshot, 123)
        self.assertEqual(meta, again)
        self.assertEqual(first._hidden, second._hidden)
        self.assertEqual(first._queue, second._queue)
        changed = deepcopy(self.snapshot)
        for order in changed._hidden.values():
            order['weight'] = 19
            order['deadline'] += 10000
        changed.initial['orders'] = [dict(o, weight=19) if o['id'] in changed._hidden else o
                                      for o in changed.initial['orders']]
        third, third_meta = resample_suffix(changed, 123)
        self.assertEqual(meta, third_meta)
        self.assertEqual(first._hidden, third._hidden)
        self.assertNotEqual(meta['suffix_hash'], resample_suffix(self.snapshot, 124)[1]['suffix_hash'])

    def test_identical_online_state_and_full_physical_prefix(self):
        original = deepcopy(self.snapshot)
        branch, meta = resample_suffix(self.snapshot, 81)
        self.assertEqual(self.snapshot.__dict__, original.__dict__)
        self.assertEqual(known_clone(branch).__dict__, known_clone(original).__dict__)
        for name in ['orders','trucks','drones','lots','ledger','trace','_actions','public']:
            self.assertEqual(getattr(branch, name), getattr(original, name), name)
        self.assertTrue(meta['online_information_unchanged'])
        self.assertFalse(known_clone(branch)._hidden)
        self.assertFalse(any(e[3] in ('request_reveal','revelation_end') for e in known_clone(branch)._queue))
        self.assertEqual({o['id'] for o in branch.initial['orders']}, set(branch.orders)|set(branch._hidden))

    def test_strict_future_slots_deadlines_and_end_order(self):
        for seed in range(6):
            branch, meta = resample_suffix(self.snapshot, seed)
            self.assertTrue(all(o['release'] > float(self.snapshot.now) for o in meta['future_orders']))
            self.assertTrue(all(o['deadline']-o['release'] in (240,300,360) for o in meta['future_orders']))
            end = [e for e in branch._queue if e[3]=='revelation_end']
            self.assertEqual(len(end),1)
            same_time = sorted(e for e in branch._queue if e[0]==end[0][0] and e[1]==1)
            self.assertEqual(same_time[-1][3], 'revelation_end')
            self.assertTrue(all(o['eligible_stations'] and 6<=o['weight']<=19 for o in branch._hidden.values()))

    def test_reject_snapshot_before_current_reveals(self):
        state = deepcopy(self.snapshot)
        one = next(iter(state._hidden.values()))
        one['release'] = state.now
        with self.assertRaisesRegex(ValueError,'after every reveal'):
            resample_suffix(state, 1)

    def test_integer_and_fractional_times_use_strictly_later_slots(self):
        # An empty realized suffix is legal under the independent slot law.
        # This isolates the slot boundary from unrelated physical timings.
        state = deepcopy(self.snapshot)
        state._hidden = {}
        state._queue = [e for e in state._queue if e[3] not in ('request_reveal','revelation_end')]
        state.now = Decimal('100')
        first, one = resample_suffix(state, 10)
        state.now = Decimal('100.999')
        second, two = resample_suffix(state, 10)
        self.assertGreater(one['count'], 0)
        self.assertEqual(one['suffix_hash'],two['suffix_hash'])
        self.assertTrue(all(o['release']>=101 for o in one['future_orders']))

    def test_resumed_full_day_reconciles_with_rewritten_initial_orders(self):
        branch, meta = resample_suffix(self.snapshot, 15)
        prefix = deepcopy(branch.trace)
        controller = deepcopy(self.controller)
        branch.start, branch._has_run = branch.now, False
        branch.run(decision_hook=controller.on_event, stop_at_terminal=True)
        self.assertTrue(branch.is_terminal())
        self.assertEqual(prefix, branch.trace[:len(prefix)])
        self.assertFalse(branch._hidden)
        initial = deepcopy(branch.initial)
        initial['stop_time'] = branch.now
        audit = audit_ledger(branch.cfg, branch.network, initial, branch.trace)
        self.assertFalse(audit['errors'])
        for name in ('revenue',)+COSTS:
            self.assertLess(abs(dec(audit['ledger'][name])-branch.ledger[name]), dec('1e-7'))
        closed, closed_meta = resample_suffix(branch, 999)
        self.assertEqual(closed_meta['count'],0)
        self.assertFalse(closed._hidden)
        self.assertFalse(any(e[3] in ('request_reveal','revelation_end') for e in closed._queue))


if __name__ == '__main__':
    unittest.main()
