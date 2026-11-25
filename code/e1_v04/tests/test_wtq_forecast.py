from copy import deepcopy
import json,random,unittest
from e1_sim.forecast import sample_releases
from e1_sim.wtq import build_controller_replay,run_controller_case
from e1_sim.runner import DEFAULT_SPEC,CONTROL_IDS,FORECAST_IDS,execute_case
from e1_sim.quantities import InvalidAction

P=json.loads(DEFAULT_SPEC.read_text(encoding='utf8'));C={c['id']:c for c in P['cases']}

class WTQForecastTests(unittest.TestCase):
    def test_original_ten_cases(self):
        for cid in sorted(CONTROL_IDS|FORECAST_IDS):
            with self.subTest(cid=cid):
                r=execute_case(P,C[cid]);self.assertEqual(r['status'],'PASS',r)

    def test_forecast_support_counts_and_seed_reproducible(self):
        public=C['E15-integer']['input']['public_information']
        for t in (300,300.5,589):
            a=sample_releases(t,public,9);b=sample_releases(t,public,9)
            self.assertEqual(a,b)
            self.assertEqual(len(a['orders']),4)
            self.assertTrue(all(t<r['release']<=590 and type(r['release']) is int for r in a['orders']))

    def test_sampler_does_not_advance_global_rng_or_mutate_public(self):
        public=deepcopy(C['E15-integer']['input']['public_information']);before=deepcopy(public)
        rng=random.getstate();sample_releases(300,public,17)
        self.assertEqual(rng,random.getstate());self.assertEqual(public,before)

    def test_public_count_inconsistency_rejected(self):
        public=deepcopy(C['E15-integer']['input']['public_information']);public['future_count']=3
        with self.assertRaises(InvalidAction):sample_releases(300,public,1)

    def test_exhausted_support_with_remaining_demand_rejected(self):
        with self.assertRaises(InvalidAction):sample_releases(590,C['E15-integer']['input']['public_information'],1)

    def test_zero_remaining_after_end_is_empty(self):
        self.assertEqual(sample_releases(700,C['E15-zero']['input']['public_information'],1)['orders'],[])

    def test_simultaneous_reveals_must_be_processed(self):
        public=deepcopy(C['E15-integer']['input']['public_information']);public['all_reveals_at_start_processed']=False
        with self.assertRaises(InvalidAction):sample_releases(300,public,1)

    def test_inherited_batch_duration_checked(self):
        case=deepcopy(C['E13-BM']);case['input']['trucks'][0]['active_operation']['end']=104
        with self.assertRaises(InvalidAction):build_controller_replay(P,case)

    def test_threshold_change_changes_measured_closing_time(self):
        case=deepcopy(C['E13-BM']);case['input']['policy']['threshold_time']=4
        s,_=run_controller_case(P,case)
        self.assertEqual(s.controller.states['T1']['closing_since'],104)
        self.assertEqual(s.controller.states['T1']['snapshot'],['P2','P3'])

    def test_snapshot_is_not_a_reservation(self):
        s,a=run_controller_case(P,C['E25'])
        self.assertTrue(a['matches_online'])
        self.assertEqual(s.controller.states['T1']['snapshot'],['P2'])
        self.assertFalse(any(e['kind']=='truck_load_started' and e['actor']=='T1' for e in s.trace))

    def test_invalid_threshold_rejected(self):
        case=deepcopy(C['E13-BM']);case['input']['policy']['threshold_weight']=0
        with self.assertRaises(InvalidAction):build_controller_replay(P,case)

    def test_known_orders_not_changed_by_threshold_observation(self):
        s=build_controller_replay(P,C['E26']);before=deepcopy(s.public_snapshot())
        s.controller.view(s,'T1');self.assertEqual(before,s.public_snapshot())

if __name__=='__main__':unittest.main()
