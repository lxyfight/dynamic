"""Single-action, shared-realization departure/wait comparisons on the frozen policy path."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
from pathlib import Path
from statistics import mean

from e1_sim.quantities import dec
from high_demand_features import FeaturePolicy, features, prediction
from recovery_opportunity import run
from run_global_demand_pilot import operation_metrics
from run_high_demand_screen import cases, OUT as PRIOR, world_for
from run_multiscale import ROOT, digest, read, write

OUT = ROOT/'tmp/wait_counterfactual_v1'
DELTA = dec(5)


def commands(plan):
    # IDs and search provenance have no physical effects. Compare all operative
    # fields, including decisions, other trucks, drones, wakeups and reviews.
    return {k:v for k,v in plan.items() if k not in ('id','enhancement')}


def new_reviews(known, plan):
    current = getattr(known,'_wait_reviews',{})
    return [(k,dec(t)) for k,t in plan.get('wait_reviews',{}).items()
            if dec(t)>known.now and dec(t)!=current.get(k)]


def pure_pairs(known, ranked, frozen):
    """Match a verified wait to the frozen immediate station departure."""
    eligible = []
    offers = [(c,tid,t) for _,_,c in ranked for tid,t in new_reviews(known,c['plan'])]
    if not offers:
        return False, eligible
    for wait,truck,review in offers:
        if review!=known.now+DELTA or truck not in known.trucks:
            continue
        vehicle = known.trucks[truck]
        station = vehicle['location']
        if station not in known.network['stations'] or vehicle['active_operation'] is not None:
            continue
        depart_trace = [e for e in frozen['trace'] if e['kind']=='road_started'
                        and e['actor']==truck and e['source']==station]
        if not depart_trace or dec(depart_trace[0]['at'])!=known.now:
            continue
        if not any(a.get('actor')==truck and a['kind']=='truck_continue_circuit'
                   and dec(a['at'])==known.now for a in frozen['plan']['actions']):
            continue
        expected = deepcopy(commands(frozen['plan']))
        for action in expected['actions']:
            if action.get('actor')==truck:
                assert dec(action['at'])>=known.now
                action['at']=dec(action['at'])+DELTA
        expected.setdefault('wait_reviews',{})[truck]=review
        if expected!=commands(wait['plan']):
            continue
        wait_trace = [e for e in wait['trace'] if e['kind']=='road_started'
                      and e['actor']==truck and e['source']==station]
        assert wait['feasible'] and wait['allowed'] and frozen['feasible'] and frozen['allowed']
        assert wait_trace and dec(wait_trace[0]['at'])==review
        eligible.append((truck,station,wait))
    return True,eligible


class PairedPolicy(FeaturePolicy):
    def __init__(self, model, target):
        super().__init__(model)
        self.target=target
        self.sites=[]
        self.chosen_sites=[]
        self.forced=0
        self.offers=0
        self.pure=0

    def pick(self, known, ranked, theta):
        plan,record=super().pick(known,ranked,theta)
        # A branch behaves identically at every other decision, also if an
        # arrival calls for replanning before the five-minute review.
        if self.target is not None and (self.forced or str(known.now)!=self.target['time']):
            return plan,record
        frozen=next(c for _,_,c in ranked if c['plan'] is plan)
        offered,matches=pure_pairs(known,ranked,frozen)
        self.offers+=int(offered)
        if not matches:
            return plan,record
        self.pure+=1
        truck,station,wait=min(matches,key=lambda row:(row[0],row[2]['id']))
        xdep,xwait=features(known,frozen),features(known,wait)
        gap_j=float(wait['score']-frozen['score'])
        gap_v=prediction(self.model,xwait)-prediction(self.model,xdep)
        site=dict(time=str(known.now),truck=truck,station=station,wait=digest(wait['plan']),
                  depart=digest(plan),prefix=digest(known.trace),offset=len(known.trace),
                  prefix_decisions=digest(self.records),candidate_count=len(ranked),
                  wait_score_delta=gap_j+gap_v,known_plan_delta=gap_j,future_value_delta=gap_v,
                  selected_feature_changes=[xwait[k]-xdep[k] for k in self.model['columns']])
        assert site['wait_score_delta']<=1e-6
        self.sites.append(site)
        if self.target is None:
            # Streaming rules depend only on past/current information.
            if not self.chosen_sites or (dec(self.chosen_sites[0]['time'])<300<=known.now
                                         and len(self.chosen_sites)==1):
                self.chosen_sites.append(site)
            return plan,record
        assert site==self.target,'branch must reach the identical public prefix and action pair'
        self.forced+=1
        record=dict(record,decision=wait['id'],plan=site['wait'],j=float(wait['score']))
        return wait['plan'],record


def inspection(state,driver,policy):
    metrics=operation_metrics(state,driver,policy)
    physical={}
    sites=policy.chosen_sites if policy.target is None else [policy.target]
    for s in sites:
        t=dec(s['time'])
        trace=state.trace[s['offset']:]
        assert digest(state.trace[:s['offset']])==s['prefix']
        roads=[e for e in trace if e['kind']=='road_started' and e['actor']==s['truck']
               and e['source']==s['station']]
        assert roads
        elapsed=float(dec(roads[0]['at'])-t)
        assert elapsed>=0
        assert elapsed==0 if policy.target is None else elapsed>0
        revealed=[o for o in state.orders.values() if o['kind']=='P' and t<dec(o['release'])<=t+DELTA]
        physical[s['time']]=dict(actual_departure_delay=elapsed,
            interval_new_pickups=len(revealed),
            interval_new_station_eligible=sum(s['station'] in o['eligible_stations'] for o in revealed))
    return dict(metrics,physical=physical)


def execute(spec,target=None):
    frozen=read(PRIOR/'freeze.json')
    world=world_for(spec)
    case_key='_'.join(map(str,spec))
    tag='depart' if target is None else 'wait_'+digest(target)[:16]
    path=OUT/'runs'/(case_key+'_'+tag+'.json')
    if path.exists():
        return read(path)
    policy=PairedPolicy(frozen['model'],target)
    row=run(world,28,policy_override=policy,metrics=inspection)
    assert policy.forced==(0 if target is None else 1)
    row.update(case=spec,input_hash=world['input_hash'],i0_hash=world['i0_hash'],target=target,
               eligible_epochs=policy.pure,offer_epochs=policy.offers,selected=policy.chosen_sites,
               sites=policy.sites)
    if target is None:
        key=digest([spec,frozen['chosen'],frozen['model'],False])
        original=read(PRIOR/'runs'/(key+'.json'))
        for k in ('net','accepted','plans','trace','ledger','input_hash','i0_hash'):
            assert original[k]==row[k],k
        assert row['offer_epochs']==row['operation_metrics']['decisions_with_new_wait']
    write(path,row)
    return row


def main():
    freeze=read(PRIOR/'freeze.json')
    assert freeze['chosen']=='shortlist_2' and len(freeze['model']['columns'])==6
    specs=cases('confirmation')
    protocol=dict(freeze=digest(freeze),cases=specs,delta=float(DELTA),
        selection='first pure matched departure per case, plus first at/after minute 300 if distinct',
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wait_rule='reserve a five minute review, permit replanning at intervening legitimate events',
        outcomes='paired realized whole-day net after one intervention, not a conditional expected value')
    p=OUT/'protocol.json'
    if p.exists():
        assert digest(read(p))==digest(protocol),'protocol changed'
    write(p,protocol)
    refs=[]
    alternatives=[]
    workers=int(os.environ.get('WAIT_CF_WORKERS','4'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(execute,sp) for sp in specs]
        branches=[]
        for f in as_completed(jobs):
            r=f.result()
            refs.append(r)
            print('depart',r['case'],r['offer_epochs'],r['eligible_epochs'],
                  [s['time'] for s in r['selected']],flush=True)
            for site in r['selected']:
                branches.append(pool.submit(execute,r['case'],site))
        for f in as_completed(branches):
            r=f.result()
            alternatives.append(r)
            print('wait',r['case'],r['target']['time'],flush=True)
    by_case={tuple(r['case']):r for r in refs}
    paired=[]
    for r in sorted(alternatives,key=lambda v:(v['case'],dec(v['target']['time']))):
        base=by_case[tuple(r['case'])]
        s=r['target']
        assert r['input_hash']==base['input_hash'] and r['i0_hash']==base['i0_hash']
        compared=dict(r['operation_metrics']['physical'][s['time']])
        ref_obs=base['operation_metrics']['physical'][s['time']]
        assert all(compared[k]==ref_obs[k] for k in ('interval_new_pickups','interval_new_station_eligible'))
        paired.append(dict(case=r['case'],time=s['time'],truck=s['truck'],station=s['station'],
            wait_minus_depart=r['net']-base['net'],accepted_delta=r['accepted']-base['accepted'],
            ledger_delta={k:r['ledger'][k]-base['ledger'][k] for k in r['ledger']},
            known_plan_delta=s['known_plan_delta'],future_value_delta=s['future_value_delta'],
            selected_feature_changes=s['selected_feature_changes'],**compared))
    assert len(paired)==sum(len(r['selected']) for r in refs)
    gains=[r['wait_minus_depart'] for r in paired]
    summary=dict(cases=len(specs),offer_epochs=sum(r['offer_epochs'] for r in refs),
        pure_frozen_departure_epochs=sum(r['eligible_epochs'] for r in refs),
        matched_pairs=len(paired),wins=sum(g>1e-7 for g in gains),losses=sum(g< -1e-7 for g in gains),
        ties=sum(abs(g)<=1e-7 for g in gains),mean_gain=mean(gains) if gains else None,
        mean_gain_per_case=mean(mean(r['wait_minus_depart'] for r in paired if r['case']==base['case'])
                                for base in refs if base['selected']) if gains else None,
        by_case=[dict(case=r['case'],offered=r['offer_epochs'],pure=r['eligible_epochs'],
                      sampled=len(r['selected'])) for r in sorted(refs,key=lambda v:v['case'])])
    write(OUT/'results.json',dict(protocol=protocol,summary=summary,paired=paired))
    print('COMPLETE',summary,flush=True)


if __name__=='__main__':
    main()
