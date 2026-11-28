"""Real physical replay verifies isolated post-unload wait interventions."""
from copy import deepcopy
import pickle
import unittest

from e1_sim.kernel import Replay
from e1_sim.online import known_clone,remaining_plan,evaluate_plan
from e1_sim.quantities import dec
from paired_wait_branches import build_branches,qualification,summarize_intervention
from test_unified_vfa_policy import two_trucks


def post_unload_state():
    original,_=two_trucks()
    cfg=dict(original.cfg,truck_handling_setup=0,drone_handling_setup_per_request=0)
    raw=deepcopy(original.initial)
    raw['trucks']=[dict(raw['trucks'][0],onboard_weight=20,unloading_phase='not_started')]
    raw['orders']=[dict(id='D'+str(i),kind='D',weight=10,release=0,deadline=100,
        city='C'+str(i+1),xy=[i*3,1],eligible_stations=['L'+str(i+1)+'+'],accepted=True)
        for i in (0,1)]
    raw['inventory']=[dict(order='D'+str(i),lot='D'+str(i),weight=10,state='onboard',holder='T1') for i in (0,1)]
    raw['actions']=[
        dict(at=0,kind='truck_unload',actor='T1',station='L1+',quantities={'D0':10}),
        dict(at=.25,kind='truck_continue_circuit',actor='T1',station='L1+'),
        dict(at=2.25,kind='truck_unload',actor='T1',station='L2+',quantities={'D1':10}),
        dict(at=2.5,kind='truck_continue_circuit',actor='T1',station='L2+'),
        dict(at=.25,kind='drone_load',actor='U1',orders=['D0']),
        dict(at=.5,kind='drone_sortie',actor='U1',visit_order=['D0']),
        dict(at=2.5,kind='drone_load',actor='U2',orders=['D1']),
        dict(at=2.75,kind='drone_sortie',actor='U2',visit_order=['D1'])]
    state=Replay(cfg,original.network,raw)
    caught=[]
    class Captured(Exception): pass
    def hook(env):
        if env.now==dec('.25'):
            caught.append(known_clone(env));raise Captured()
    try:state.run(decision_hook=hook)
    except Captured:pass
    known=caught[0]
    row=evaluate_plan(known,remaining_plan(known),'WK')
    assert row['feasible'],row.get('reason')
    return known,row


class PairedWaitBranchTests(unittest.TestCase):
    def test_real_unload_four_feasible_branches_freeze_routes_and_retime_dependencies(self):
        known,base=post_unload_state()
        before=pickle.dumps(known)
        rows=build_branches(known,base,'T1')
        self.assertEqual(set(rows),{0,2,5,10})
        self.assertEqual(pickle.dumps(known),before)
        for h,row in rows.items():
            self.assertTrue(row['state'].is_terminal())
            self.assertEqual(row['score']-rows[0]['score'],-dec(h)*dec(known.cfg['truck_time_cost']))
            self.assertEqual(row['plan']['decisions'],base['plan']['decisions'])
            changes=row['intervention']['timing_only_changes']
            if h:
                self.assertTrue(any(c['actor']=='U2' and c['delay']==h for c in changes))
                self.assertFalse(any(c['actor']=='U1' for c in changes))
                self.assertFalse(row['intervention']['strict_non_target_actions_unchanged'])

    def test_non_unload_and_reserved_loading_are_not_active_departure_states(self):
        known,base=post_unload_state()
        known.trace=[e for e in known.trace if e['kind']!='truck_unload_done']
        self.assertIn('not_just_unloaded',qualification(known,base,'T1')['reasons'])
        with self.assertRaises(ValueError):build_branches(known,base,'T1')
        self.assertTrue(qualification(known,base,'T1',require_just_unloaded=False)['eligible'])
        self.assertEqual(set(build_branches(known,base,'T1',require_just_unloaded=False)),{0,2,5,10})
        known,base=post_unload_state()
        reserved=deepcopy(base)
        reserved['plan']['actions'][0]['kind']='truck_load'
        result=qualification(known,reserved,'T1')
        self.assertTrue(result['known_cargo_reservation'])
        self.assertFalse(result['eligible'])
        self.assertFalse(qualification(known,reserved,'T1',require_just_unloaded=False)['eligible'])

    def test_existing_timer_and_changed_assignment_are_rejected(self):
        known,base=post_unload_state()
        known._wait_reviews={'T1':known.now+5}
        self.assertIn('existing_active_review',qualification(known,base,'T1')['reasons'])
        known._wait_reviews={}
        changed=deepcopy(base['plan'])
        next(a for a in changed['actions'] if a['kind']=='drone_sortie')['visit_order']=['D1']
        with self.assertRaisesRegex(ValueError,'assignment'):
            summarize_intervention(known,base['plan'],changed,'T1',0)


if __name__=='__main__':unittest.main()
