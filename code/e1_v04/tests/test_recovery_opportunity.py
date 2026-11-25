from copy import deepcopy
from types import SimpleNamespace
import unittest

from e1_sim.information import expected_arrivals
from e1_sim.online import known_clone,prepare_online
from e1_sim.quantities import dec
from recovery_opportunity import RecoveryPolicy,coverage
from test_unknown_demand import CFG,instance


class RecoveryOpportunityTests(unittest.TestCase):
    def test_union_and_incremental_coverage_only(self):
        law=instance()['input']['public_information']
        world=SimpleNamespace(public=law,now=10,network={'stations':{'L1+':{},'L1-':{},'L2+':{}}})
        plan={'actions':[dict(kind='truck_dispatch',actor=k,delivery_orders=[]) for k in ('one','two')]}
        trace=[dict(kind='road_started',actor=k,source=s,at=t) for k,s,t in [
            ('active','L1+',30),('one','L1+',50),('two','L1-',60),
            ('one','L2+',595),('active','L2+',580)]]
        c=dict(trace=trace,plan=plan)
        self.assertAlmostEqual(coverage(world,c),expected_arrivals(law,'C1',30,60)+expected_arrivals(law,'C2',580,590))
        self.assertEqual(coverage(world,dict(trace=trace,plan={'actions':[]})),0.)
        world.public=dict(law,revelation_closed=True)
        self.assertEqual(coverage(world,c),0.)

    def test_new_candidate_is_common_feasible_and_nonanticipatory(self):
        # Reveal one isolated pickup after initial trucks have returned. It must
        # have an actual depot trip. Alter the hidden suffix in the second world.
        initial=instance()
        known_states=[]
        for extra in (False,True):
            x=deepcopy(initial);raw=x['input']
            p=deepcopy(next(o for o in raw['orders'] if o['kind']=='P'))
            p.update(release=240,deadline=600,id='observed')
            raw['orders']=[o for o in raw['orders'] if o['kind']=='D']+[p]
            raw['external_events']=[dict(kind='request_reveal',at=240,orders=['observed'])]
            if extra:
                future=dict(p,release=500,deadline=860,id='later')
                raw['orders'].append(future);raw['external_events'].append(dict(kind='request_reveal',at=500,orders=['later']))
            state,driver=prepare_online(raw,CFG,x['network'],full_day=True)
            def hook(s):
                if s.now==240:
                    known_states.append(known_clone(s))
                return driver.on_event(s)
            driver.policy=RecoveryPolicy(0);state.full_day=False;state.stop=dec(240);driver.extend_horizon=False
            state.run(decision_hook=hook)
        self.assertEqual(len(known_states),2)
        rankings=[]
        for state,w in zip(known_states,(0,48)):
            policy=RecoveryPolicy(w)
            evaluated,_,_=policy.propose(state)
            ranked=policy.rank(state,evaluated)
            deferred=[(j,phi,p) for j,phi,p in ranked if '~defer_' in p['id']]
            self.assertTrue(deferred)
            for j,phi,p in deferred:
                parent=next(r for r in evaluated if r['id']==p['id'].split('~defer_')[0])
                cmd=lambda row:next(a for a in row['plan']['actions'] if a['kind']=='truck_dispatch')
                self.assertGreater(dec(cmd(p)['at']),dec(cmd(parent)['at']))
                self.assertLessEqual(dec(cmd(p)['at']),dec(CFG['truck_dispatch_cutoff']))
                self.assertTrue(p['allowed'] and p['feasible'])
                self.assertLessEqual(p['state'].orders['observed']['completed_at'],dec(600)+dec('1e-7'))
                self.assertAlmostEqual(float(j),float(parent['score']),places=6)
            rankings.append([(p['id'],j,phi) for j,phi,p in ranked])
        self.assertEqual(*rankings)
