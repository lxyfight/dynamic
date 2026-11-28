"""Joint-action ranking and fixed, truck-local waiting clocks."""
from copy import deepcopy
import json
import pickle
import unittest
from unittest.mock import patch

from e1_sim.cost_model import current_configuration
from e1_sim.kernel import Replay
from e1_sim.online import (_new_truck, evaluate_plan, install_plan, known_clone,
                           pickup_variants, prepare_online, remaining_plan)
from e1_sim.quantities import dec
from e1_sim.runner import DEFAULT_SPEC
from e1_sim.unknown_demand import public_law
from unified_vfa_features import FEATURE_NAMES
from unified_vfa_policy import MODEL_VERSION, UnifiedVFAPolicy, joint_wait_pool, operating_rules


def two_trucks():
    cfg = current_configuration(json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))['configuration'])
    net = dict(stations={'L1+': [0, 0], 'L2+': [3, 0]}, arcs=[
        dict(**{'from': 'depot', 'to': 'L1+'}, km=2, min=2, tolled=True),
        dict(**{'from': 'L1+', 'to': 'L2+'}, km=2, min=2, tolled=True),
        dict(**{'from': 'L2+', 'to': 'depot'}, km=2, min=2, tolled=True)])
    trucks = [dict(id=tid, location=station, onboard_weight=0, departed_at=0,
                   unloading_phase='skipped', completed_batches=0, active_operation=None)
              for tid, station in [('T1', 'L1+'), ('T2', 'L2+')]]
    drones = [dict(id=uid, home=s, location=s, activated=False, onboard_weight=0,
                   takeoff_at=None, active_operation=None)
              for uid, s in [('U1', 'L1+'), ('U2', 'L2+')]]
    law = public_law(2, 4, end=30)
    law['demand_model']['spatial_service_lags_by_city'] = {'C1': [1., 2.], 'C2': [1., 2.]}
    raw = dict(start_time=0, stop_time=100, orders=[], trucks=trucks, drones=drones,
               inventory=[], actions=[], external_events=[], public_information=law)
    known = Replay(cfg, net, raw)
    plan = dict(id='base', decisions={}, new_trucks=[], wait_reviews={}, actions=[
        dict(at=0, kind='truck_continue_circuit', actor=t['id'], station=t['location']) for t in trucks])
    return known, plan


def model(h=5, weight=1.):
    cfg = current_configuration(json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))['configuration'])
    return dict(h=h, version=MODEL_VERSION, operating_rules=operating_rules(cfg, (360,)),
                feature_names=list(FEATURE_NAMES),
                weights=[weight]+[0.]*(len(FEATURE_NAMES)-1), scales=[1.]*len(FEATURE_NAMES))


class UnifiedPolicyTests(unittest.TestCase):
    def test_joint_wait_keeps_every_base_and_two_trucks_can_wait(self):
        known, plan = two_trucks()
        base = evaluate_plan(known, plan)
        before = pickle.dumps(known)
        pool, audit = joint_wait_pool(known, [base], 5)
        self.assertEqual(len(pool), 4)
        self.assertIs(pool[0], base)
        joint = next(c for c in pool if len(c['plan']['wait_reviews']) == 2)
        self.assertEqual(joint['plan']['wait_reviews'], {'T1': dec(5), 'T2': dec(5)})
        self.assertTrue(joint['state'].is_terminal())
        self.assertEqual(joint['score']-base['score'], -dec(10)*dec(known.cfg['truck_time_cost']))
        self.assertEqual(before, pickle.dumps(known))
        self.assertEqual(audit['attempted'], 3)
        self.assertFalse(audit['truncated'])

    def test_one_scorer_can_choose_wait_on_inferior_base_plan(self):
        known, plan = two_trucks()
        base = evaluate_plan(known, plan)
        alternative = deepcopy(plan)
        alternative['id'] = 'costlier_base'
        alternative['new_trucks'] = [_new_truck('EXTRA')]
        alternative['actions'].append(dict(at=0, kind='truck_dispatch', actor='EXTRA', delivery_orders=[]))
        other = evaluate_plan(known, alternative)
        self.assertTrue(other['feasible'], other.get('reason'))
        self.assertLess(other['score'], base['score'])
        def controlled_feature(state, row, h):
            # A deliberately prescribed future value tests joint maximization;
            # all known-task J values still come from the real physical oracle.
            opportunity = 10000. if row['plan']['new_trucks'] and len(row['plan']['wait_reviews']) == 2 else 0.
            return (opportunity,) + (0.,)*(len(FEATURE_NAMES)-1)
        policy = UnifiedVFAPolicy(model=model(), enhancement_budget=0)
        with patch('unified_vfa_policy.features', side_effect=controlled_feature):
            chosen = policy.choose(known, [base, other])
        self.assertEqual(len(chosen['plan']['wait_reviews']), 2)
        self.assertEqual(chosen['plan']['new_trucks'][0]['id'], 'EXTRA')
        self.assertEqual(len(policy.selection_log[-1]['pool']), 8)
        self.assertEqual([r['base_id'] for r in policy.selection_log[-1]['waiting']['by_base']],
                         ['base', 'costlier_base'])

    def test_zero_model_is_exact_myopic_over_same_joint_pool(self):
        known, plan = two_trucks()
        base = evaluate_plan(known, plan)
        policy = UnifiedVFAPolicy(h=5, model=None, enhancement_budget=0)
        self.assertEqual(policy.choose(known, [base])['plan'], base['plan'])
        self.assertTrue(all(r['W'] == 0 for r in policy.selection_log[-1]['pool']))
        self.assertEqual(policy.selection_log[-1]['model_mode'], 'zero_residual')
        self.assertTrue(all('y' not in sample for sample in policy.samples))

    def test_budget_is_per_base_and_reports_omitted_combinations(self):
        known, plan = two_trucks()
        a = evaluate_plan(known, plan)
        plan = deepcopy(plan)
        plan['id'] = 'second_base'
        b = evaluate_plan(known, plan)
        rows, stats = joint_wait_pool(known, [a, b], 5, budget_per_base=1)
        self.assertEqual(len(rows), 4)
        self.assertTrue(stats['truncated'])
        self.assertEqual(stats['attempted'], 2)
        self.assertTrue(all(len(c['plan']['wait_reviews']) == 2 for c in rows[2:]))
        self.assertTrue(all(r['attempted'] == 1 and r['combinations'] == 3 for r in stats['by_base']))

    def test_early_event_keeps_timer_and_elapsed_wait_then_renews_at_expiry(self):
        known, plan = two_trucks()
        base = evaluate_plan(known, plan)
        joint = next(c for c in joint_wait_pool(known, [base], 5)[0]
                     if len(c['plan']['wait_reviews']) == 2)
        install_plan(known, joint['plan'])
        known._advance(dec(2))
        keep = evaluate_plan(known, remaining_plan(known))
        rows, _ = joint_wait_pool(known, [keep], 5)
        self.assertTrue(all(all(t == 5 for t in row['plan']['wait_reviews'].values()) for row in rows))
        self.assertTrue(any(not row['plan']['wait_reviews'] for row in rows))
        install_plan(known, keep['plan'])
        self.assertEqual(known._station_wait_memory['T1']['review_count'], 1)
        self.assertEqual(known._station_wait_memory['T1']['active_wait_minutes'], 2)
        known._advance(dec(5))
        due = evaluate_plan(known, remaining_plan(known))
        renewed = next(c for c in joint_wait_pool(known, [due], 5)[0]
                       if c['plan']['wait_reviews'].get('T1') == 10 and c['plan']['wait_reviews'].get('T2') == 10)
        install_plan(known, renewed['plan'])
        self.assertEqual(known._station_wait_memory['T1']['review_count'], 2)
        self.assertEqual(known._station_wait_memory['T1']['active_wait_minutes'], 5)

    def test_same_h_independent_start_times_and_no_global_calendar(self):
        known, plan = two_trucks()
        plan['actions'][0]['at'] = 5
        plan['wait_reviews'] = {'T1': dec(5)}
        install_plan(known, plan)
        known._advance(dec(2))
        # T2 obtains its own decision at t=2; T1's earlier timer remains due at5.
        plan = remaining_plan(known)
        plan['actions'].append(dict(at=2, kind='truck_continue_circuit', actor='T2', station='L2+'))
        base = evaluate_plan(known, plan)
        self.assertTrue(base['feasible'], base.get('reason'))
        rows, _ = joint_wait_pool(known, [base], 5)
        joint = next(c for c in rows if c['plan']['wait_reviews'] == {'T1': dec(5), 'T2': dec(7)})
        install_plan(known, joint['plan'])
        self.assertEqual(known._station_wait_memory['T1']['review_count'], 1)
        self.assertEqual(known._station_wait_memory['T2']['review_count'], 1)

    def test_real_request_event_replans_without_restarting_wait_clock(self):
        known, plan = two_trucks()
        raw = deepcopy(known.initial)
        raw['actions'] = plan['actions']
        raw['orders'] = [dict(id='P1', kind='P', weight=6, release=2, deadline=100,
            accepted=None, city='C1', xy=[0, 1], eligible_stations=['L1+'])]
        raw['external_events'] = [dict(at=2, kind='request_reveal', orders=['P1'])]
        state, driver = prepare_online(raw, known.cfg, known.network)
        driver.policy = UnifiedVFAPolicy(h=5, model=model(weight=10000),
                                         enhancement_budget=0, candidate_budget=2)
        at_two = []
        def hook(env):
            driver.on_event(env)
            if env.now == 2:
                at_two.append((deepcopy(env._wait_reviews), deepcopy(env._station_wait_memory)))
        def prefer_keeping_clocks(state, row, h):
            return (float(len(row['plan'].get('wait_reviews', {}))),)+(0.,)*(len(FEATURE_NAMES)-1)
        with patch('unified_vfa_policy.features', side_effect=prefer_keeping_clocks):
            state.run(decision_hook=hook, stop_at_terminal=True)
        self.assertTrue(state.is_terminal())
        self.assertEqual(len(at_two), 1)
        reviews, memories = at_two[0]
        self.assertEqual(reviews, {'T1': dec(5), 'T2': dec(5)})
        self.assertEqual(memories['T1']['review_count'], 1)
        self.assertEqual(memories['T1']['active_wait_minutes'], 2)
        self.assertEqual(state._driver_calls[dec(2)], 1)
        self.assertTrue(all(not c._hidden for c in [known_clone(state)]))

    def test_known_cargo_reservation_is_not_speculative_wait(self):
        known, plan = two_trucks()
        known.orders['P1'] = dict(id='P1', kind='P', weight=6, release=dec(0), deadline=dec(100),
            accepted=True, city='C1', xy=[0, 1], eligible_stations=['L1+'], completed_at=None,
            available_at=dec(0), collected_at=dec(0), station='L1+', assigned_drone=None)
        known._new_lot('P1', 6, 'available', 'L1+')
        plan['actions'][0]['at'] = dec('8.15')
        plan['actions'].insert(0, dict(at=8, kind='truck_load', actor='T1', station='L1+', orders=['P1']))
        base = evaluate_plan(known, plan)
        self.assertTrue(base['feasible'], base.get('reason'))
        rows, _ = joint_wait_pool(known, [base], 5)
        for row in rows:
            self.assertNotIn('T1', row['plan']['wait_reviews'])
            self.assertEqual(next(a['at'] for a in row['plan']['actions'] if a['kind'] == 'truck_load'), 8)

    def test_positive_fixed_h_is_not_clipped_near_end_and_ids_include_h(self):
        known, plan = two_trucks()
        known.public['revelation_end'] = 3
        for segments in known.public['demand_model']['rates_by_city'].values():
            segments[0]['end'] = 3
        base = evaluate_plan(known, plan)
        a, _ = joint_wait_pool(known, [base], 5)
        b, _ = joint_wait_pool(known, [base], 2)
        self.assertEqual(a[1]['plan']['wait_reviews']['T1'], 5)
        self.assertNotEqual(a[1]['id'], b[1]['id'])
        for h in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                UnifiedVFAPolicy(h=h)

    def test_model_schema_h_scales_and_legacy_weights_are_rejected(self):
        for bad in ({'columns': [], 'weights': []}, model(h=10),
                    dict(model(), scales=[0]*len(FEATURE_NAMES)),
                    dict(model(), weights=[float('nan')]*len(FEATURE_NAMES)),
                    dict(model(), feature_names=list(reversed(FEATURE_NAMES)))):
            with self.assertRaises(ValueError):
                UnifiedVFAPolicy(h=5, model=bad)
        policy = UnifiedVFAPolicy(h=5)
        with self.assertRaises(AttributeError):
            policy.h = 10

    def test_active_drone_loading_allows_planned_future_sortie_insertion(self):
        known, plan = two_trucks()
        known.orders['D1'] = dict(id='D1', kind='D', weight=6, release=dec(0), deadline=dec(100),
            accepted=True, city='C1', xy=[0, 1], eligible_stations=['L1+'], completed_at=None,
            available_at=dec(0), collected_at=None, station='L1+', assigned_drone=None)
        known._new_lot('D1', 6, 'available', 'L1+')
        known.apply(dict(kind='drone_load', actor='U1', orders=['D1']))
        finish = known.drones['U1']['active_operation']['end']
        plan['actions'].append(dict(at=finish, kind='drone_sortie', actor='U1', visit_order=['D1']))
        known.orders['P1'] = dict(id='P1', kind='P', weight=6, release=dec(0), deadline=dec(100),
            accepted=True, city='C1', xy=[0, 2], eligible_stations=['L1+'], completed_at=None,
            available_at=None, collected_at=None, station=None, assigned_drone=None)
        known._new_lot('P1', 6, 'customer', 'P1')
        before = pickle.dumps(known)
        inserted = [evaluate_plan(known, p) for p in pickup_variants(known, plan, 'P1')
                    if any(a['kind'] == 'drone_sortie' and set(a['visit_order']) == {'D1', 'P1'}
                           for a in p['actions'])]
        self.assertTrue(any(c['feasible'] for c in inserted))
        self.assertEqual(before, pickle.dumps(known))
        self.assertIsNotNone(known.drones['U1']['active_operation'])

        # The same drone becomes immutable at actual takeoff. New pickup work
        # can still be added in a later sortie after this committed return.
        known.stop = finish
        known.run()
        known.apply(dict(kind='drone_sortie', actor='U1', visit_order=['D1']))
        known = known_clone(known)
        suffix = dict(id='airborne', decisions={}, new_trucks=[], wait_reviews={}, actions=[
            dict(at=known.now, kind='truck_continue_circuit', actor=t['id'], station=t['location'])
            for t in known.trucks.values()])
        before = pickle.dumps(known)
        later = [evaluate_plan(known, p) for p in pickup_variants(known, suffix, 'P1')]
        self.assertTrue(any(c['feasible'] for c in later))
        self.assertTrue(all('D1' not in a['visit_order'] for c in later for a in c['plan']['actions']
                            if a['kind'] == 'drone_sortie'))
        self.assertEqual(known.drones['U1']['sortie_commitment']['visit_order'], ['D1'])
        self.assertEqual(before, pickle.dumps(known))


if __name__ == '__main__':
    unittest.main()
