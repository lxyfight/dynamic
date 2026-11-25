"""Enumerate every pure waiting/departure decision on eight fixed policy paths."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from statistics import mean
from time import monotonic,sleep
import hashlib
import os

from run_wait_counterfactual import (OUT, PRIOR, DELTA, cases, world_for, run,
    read, write, digest, FeaturePolicy, pure_pairs, new_reviews, execute, dec)

DEST = OUT/'all_eligible'


class OfferAudit(FeaturePolicy):
    def __init__(self, model):
        super().__init__(model)
        self.audit=[]

    def pick(self, known, ranked, theta):
        plan,record=super().pick(known,ranked,theta)
        frozen=next(c for _,_,c in ranked if c['plan'] is plan)
        offered,matches=pure_pairs(known,ranked,frozen)
        if not offered:
            return plan,record
        all_waits=[(c,actor,rev) for _,_,c in ranked for actor,rev in new_reviews(known,c['plan'])]
        # If the frozen action lacks a match, test whether another feasible
        # departure in the same pool can be paired to a physically identical wait.
        any_alternative=False
        if not matches:
            any_alternative=any(pure_pairs(known,ranked,c)[1] for _,_,c in ranked if c is not frozen)
        first_actions=set()
        for c,actor,rev in all_waits:
            own=sorted((a for a in c['plan']['actions'] if a.get('actor')==actor),key=lambda a:dec(a['at']))
            if own:
                first_actions.add((own[0]['kind'],dec(own[0]['at'])==rev))
        if matches:
            status='matched_frozen_departure'
        elif any_alternative:
            status='matched_other_departure_only'
        elif first_actions=={('truck_load',True)}:
            status='all_waits_postpone_loading'
        else:
            status='other_unmatched_or_nonisolated'
        self.audit.append(dict(time=str(known.now),frozen_id=frozen['id'],status=status,
            matched_trucks=sorted({truck for truck,_,_ in matches}),
            station=matches[0][1] if matches else None,
            option_first_actions=sorted(first_actions),
            frozen_due=[(a['actor'],a['kind']) for a in plan['actions']
                        if dec(a['at'])==known.now and a['kind'].startswith('truck_')]))
        return plan,record


def observe_departures(state,driver,policy):
    for a in policy.audit:
        if a['status']!='matched_frozen_departure':continue
        tid,=a['matched_trucks']
        t=dec(a['time'])
        assert any(e['kind']=='road_started' and e['actor']==tid and e['source']==a['station']
                   and dec(e['at'])==t for e in state.trace)
        a['actual_departed_at']=float(t)
        revealed=[o for o in state.orders.values() if o['kind']=='P' and t<dec(o['release'])<=t+DELTA]
        a['interval_new_pickups']=len(revealed)
        a['interval_new_station_eligible']=sum(a['station'] in o['eligible_stations'] for o in revealed)
    return {}


def run_audit(spec):
    path=DEST/'audit'/('_'.join(map(str,spec))+'.json')
    if path.exists():return read(path)
    p=OfferAudit(read(PRIOR/'freeze.json')['model'])
    value=run(world_for(spec),28,policy_override=p,metrics=observe_departures)
    baseline=read(OUT/'runs'/('_'.join(map(str,spec))+'_depart.json'))
    assert all(value[k]==baseline[k] for k in ('net','ledger','trace','plans','accepted'))
    matched=[r for r in p.audit if r['status']=='matched_frozen_departure']
    assert len(p.audit)==baseline['offer_epochs'] and len(matched)==baseline['eligible_epochs']
    assert [s['time'] for s in matched]==[s['time'] for s in baseline['sites']]
    assert all(len(r['matched_trucks'])==1 for r in matched),'multiple truck targets at one event'
    write(path,p.audit)
    return p.audit


def main():
    specs=cases('confirmation')
    first=read(OUT/'protocol.json')
    protocol=dict(base_protocol=digest(first),scope='all exact matched epochs on the eight frozen paths',
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampling='One target truck per matched epoch verified; every branch shares the original suffix; events in the same scenario are dependent.')
    if (DEST/'protocol.json').exists():assert digest(read(DEST/'protocol.json'))==digest(protocol)
    write(DEST/'protocol.json',protocol)
    started=monotonic()
    totals=[]
    pending=list(specs)
    jobs=[]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('ALL_WAIT_WORKERS',4))) as pool:
        while pending:
            for spec in list(pending):
                path=OUT/'runs'/('_'.join(map(str,spec))+'_depart.json')
                if not path.exists():continue
                base=read(path)
                totals.append(base)
                pending.remove(spec)
                jobs.extend(pool.submit(execute,spec,s) for s in base['sites'] if s not in base['selected'])
                jobs.append(pool.submit(run_audit,spec))
            if pending:sleep(1)
            if monotonic()-started>3600:raise TimeoutError('waiting for reference paths')
        for i,f in enumerate(as_completed(jobs),1):
            f.result()
            print('all-eligible additional job',i,'/',len(jobs),flush=True)
    paired=[]
    audits=[]
    for base in sorted(totals,key=lambda r:r['case']):
        spec=base['case']
        audit=read(DEST/'audit'/('_'.join(map(str,spec))+'.json'))
        audits.extend(dict(case=spec,**r) for r in audit)
        for s in base['sites']:
            path=OUT/'runs'/('_'.join(map(str,spec))+'_wait_'+digest(s)[:16]+'.json')
            # The first-stage run owns these files for its predefined subset.
            while not path.exists():
                if monotonic()-started>3600:raise TimeoutError('missing primary branch')
                sleep(1)
            alt=read(path)
            assert alt['case']==spec and alt['target']==s
            assert alt['input_hash']==base['input_hash'] and alt['i0_hash']==base['i0_hash']
            assert alt['sites']==[s]
            physically=alt['operation_metrics']['physical'][s['time']]
            assert physically['actual_departure_delay']>0
            actual=next(r for r in audit if r['time']==s['time'])
            assert all(actual[k]==physically[k] for k in ('interval_new_pickups','interval_new_station_eligible'))
            if s in base['selected']:
                observed=base['operation_metrics']['physical'][s['time']]
                assert observed['actual_departure_delay']==0
                assert observed['interval_new_pickups']==physically['interval_new_pickups']
            deltas={k:alt['ledger'][k]-base['ledger'][k] for k in base['ledger']}
            gain=alt['net']-base['net']
            assert abs(gain-(deltas['revenue']-sum(v for k,v in deltas.items() if k not in ('revenue','net'))))<1e-6
            paired.append(dict(case=spec,time=s['time'],truck=s['truck'],station=s['station'],
                predefined=s in base['selected'],wait_minus_depart=gain,depart_net=base['net'],wait_net=alt['net'],
                departed_at=actual['actual_departed_at'],waited_departed_at=float(s['time'])+physically['actual_departure_delay'],
                accepted_delta=alt['accepted']-base['accepted'],ledger_delta=deltas,**physically,
                delta_j=s['known_plan_delta'],delta_v=s['future_value_delta'],score_delta=s['wait_score_delta'],
                feature_differences=s['selected_feature_changes']))
    assert len(audits)==53 and len(paired)==sum(r['eligible_epochs'] for r in totals)
    rows=[r['wait_minus_depart'] for r in paired]
    case_means=[mean(r['wait_minus_depart'] for r in paired if r['case']==b['case']) for b in totals]
    delta_vs=[r['delta_v'] for r in paired]
    summary=dict(cases=len(totals),offer_epochs=len(audits),pure_pairs=len(paired),
        offer_reasons=dict(Counter(r['status'] for r in audits)),
        wins=sum(v>1e-7 for v in rows),ties=sum(abs(v)<=1e-7 for v in rows),losses=sum(v< -1e-7 for v in rows),
        mean_gain=mean(rows),mean_gain_equal_case=mean(case_means),range=[min(rows),max(rows)],
        delta_v_positive=sum(v>1e-7 for v in delta_vs),delta_v_zero=sum(abs(v)<=1e-7 for v in delta_vs),
        delta_v_negative=sum(v< -1e-7 for v in delta_vs),
        feature_change_counts=[sum(abs(r['feature_differences'][i])>1e-7 for r in paired) for i in range(6)],
        per_case=[dict(case=b['case'],pure=b['eligible_epochs'],offer=b['offer_epochs'],
                       mean_gain=g) for b,g in zip(totals,case_means)])
    write(DEST/'results.json',dict(protocol=protocol,summary=summary,paired=paired,offer_audit=audits))
    print('COMPLETE',summary,flush=True)


if __name__=='__main__':
    main()
