"""No full-day runs: gate guards and real known-only four-action branches."""
from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from fit_paired_chain import VERSION,write
import run_paired_chain_full_day as full
from test_paired_wait_branches import post_unload_state
from unified_vfa_features import FEATURE_NAMES
from unified_vfa_policy import canonical_plan_digest


def model():
    return dict(version=VERSION,kind='paired',columns=[0,1],intercept=0.,
        feature_names=list(FEATURE_NAMES[:2]),weights=[1.,1.],scales=[1.,1.])


def gate_fixture():
    value=model()
    inference={ref:dict(n=8,t_df=7,ci95=[.1,2.]) for ref in ('depart','fixed10')}
    result=dict(stage='augmented',dev_evaluated=True,primary_family='paired_chain',
        proceed_to_full_day_model_comparison=True,
        candidate_gates=dict(paired_chain=dict(pass_both_lower95=True,complete_eight_layouts=True,
            selected_primary=True,authoritative_for_full_day=True)),
        reports=dict(dev=dict(models=dict(paired_chain=dict(layout_inference=inference)))))
    frozen=dict(primary_family='paired_chain',selected_models=dict(paired_chain=value))
    return result,frozen,value


class PairedChainFullDayTests(unittest.TestCase):
    def test_both_manual_stages_reject_false_gate_before_any_world_or_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            write(output/'augmented/results.json',dict(proceed_to_full_day_model_comparison=False))
            with patch.object(full,'make_world') as world,patch.object(full,'execute_jobs') as jobs:
                for stage in ('tune-fixed','dev'):
                    with self.assertRaisesRegex(ValueError,'gate is false'):
                        full.run(output,stage)
                world.assert_not_called();jobs.assert_not_called()
            self.assertFalse((output/'full_day').exists())

    def test_gate_recomputes_both_intervals_and_checks_frozen_model(self):
        result,frozen,value=gate_fixture()
        self.assertEqual(full.validate_action_gate(result,frozen,value),'paired_chain')
        bad=deepcopy(result)
        bad['reports']['dev']['models']['paired_chain']['layout_inference']['fixed10']['ci95'][0]=0.
        with self.assertRaisesRegex(ValueError,'lower95'):
            full.validate_action_gate(bad,frozen,value)
        altered=deepcopy(value);altered['weights'][0]=2.
        with self.assertRaisesRegex(ValueError,'tune-frozen'):
            full.validate_action_gate(result,frozen,altered)

    def test_adaptive_and_fixed_menus_share_depart_and_model_scores(self):
        scores=[0,4,3,8]
        self.assertEqual(full.action_menu('adaptive'),(0,2,5,10))
        self.assertEqual(full.action_menu('fixed',5),(0,5))
        self.assertEqual(full.menu_choice(scores,full.action_menu('adaptive')),10)
        self.assertEqual(full.menu_choice(scores,full.action_menu('fixed',5)),5)
        self.assertEqual(full.menu_choice([0,-2,-3,-5],(0,10)),0)
        self.assertEqual(full.menu_choice([0,1e-10,0,0],(0,2)),0)
        with self.assertRaises(ValueError):full.action_menu('fixed',3)

    def test_same_none_base_and_physical_branches_preserve_assignments(self):
        known,base=post_unload_state()
        before=pickle.dumps(known)
        base_hash=canonical_plan_digest(base['plan'])
        policy=full.PairedReviewPolicy(model(),'fixed',5,action_solver='bounded',
            enhancement_budget=0,wait_budget_per_base=0,candidate_budget=1)
        with patch.object(policy,'score_branches',return_value=([0,2,5,10],[[0.]*7 for _ in range(4)],[])):
            chosen=policy.choose(known,[base])
        event=policy.selector_log[-1]
        self.assertEqual(event['none_plan_hash'],base_hash)
        self.assertEqual(event['selected_h'],5)
        self.assertEqual(event['menu'],[0,5])
        self.assertTrue(event['all_four_verified'])
        self.assertEqual(set(event['branch_audits']),{'0','2','5','10'})
        self.assertEqual(chosen['plan']['decisions'],base['plan']['decisions'])
        self.assertTrue(all(a['base_hash']==base_hash and a['admissions_unchanged']
                            and a['action_identities_unchanged'] for a in event['branch_audits'].values()))
        self.assertEqual(pickle.dumps(known),before)
        self.assertIsNone(policy.model)
        self.assertEqual(policy.selection_log[-1]['canonical_plan_hash'],canonical_plan_digest(chosen['plan']))
        self.assertEqual(policy.selection_log[-1]['chosen_wait_intervals'],{'T1':5.})
        self.assertEqual(policy.records[-1]['decision'],chosen['id'])
        self.assertEqual(policy.samples[-1]['plan'],policy.selection_log[-1]['plan_hash'])
        self.assertEqual(policy.search_metrics()['chosen_wait_interval_counts'],{'5':1})
        self.assertEqual(policy.selector_metrics()['active_wait_interval_counts'],{5:1})

    def test_multiple_eligible_trucks_fall_back_without_branch_intervention(self):
        known,base=post_unload_state()
        known.trucks['T2']=deepcopy(known.trucks['T1'])
        policy=full.PairedReviewPolicy(model(),'adaptive',action_solver='bounded')
        def qualify(known,base,truck,require_just_unloaded):
            return dict(truck=truck,eligible=True,just_unloaded=True)
        with patch.object(full.UnifiedVFAPolicy,'choose',return_value=base), \
             patch.object(full,'qualification',side_effect=qualify), \
             patch.object(full,'build_branches') as build:
            self.assertIs(policy.choose(known,[base]),base)
            build.assert_not_called()
        self.assertEqual(policy.selector_log[-1]['reason'],'multiple_eligible_trucks')

    def test_fixed_h_selection_uses_equal_tune_days_and_shorter_tie(self):
        rows=[dict(case='c'+str(i),policy='fixed'+str(h),net=10 if h in (5,10) else 9)
              for i in range(4) for h in (2,5,10)]
        result=full.select_fixed(rows)
        self.assertEqual(result['selected_h'],5)
        self.assertFalse(result['dev_returns_used'])
        with self.assertRaises(ValueError):full.select_fixed(rows[:-1])


if __name__=='__main__':
    unittest.main()
