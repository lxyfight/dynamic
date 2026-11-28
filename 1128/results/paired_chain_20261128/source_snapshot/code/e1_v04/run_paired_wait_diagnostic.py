"""Offline conditional, paired one-truck review interventions on none paths."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter

from e1_sim.audit import audit_ledger
from e1_sim.constraints import distance
from e1_sim.initializer import digest
from e1_sim.kernel import COSTS, require
from e1_sim.online import known_clone, prepare_online, OnlineCoordinator
from e1_sim.quantities import dec, plain
from run_feature_screen import SEARCH, SETTING, WORLD_OPTIONS, write
from unified_vfa_policy import UnifiedVFAPolicy, canonical_plan_digest
from uniform_wait_worlds import make_world

PRIOR=Path('../../results/feature_screen_20261128')
DEFAULT_OUT=Path('../../results/paired_wait_20261128')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def save_pickle(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(str(p),'wb') as f: pickle.dump(obj,f,protocol=4)


def load_pickle(p):
    with gzip.open(str(p),'rb') as f: return pickle.load(f)


def final_metrics(state):
    require(state.is_terminal(),'nonterminal branch')
    require(all(o['completed_at'] is not None and o['completed_at']<=o['deadline']
        for o in state.orders.values() if o['kind']=='D'),'missed mandatory delivery')
    pickups=[o for o in state.orders.values() if o['kind']=='P']
    require(all(o['accepted'] is not None and (not o['accepted'] or o['completed_at'] is not None)
        for o in pickups),'unresolved pickup')
    initial=deepcopy(state.initial);initial['stop_time']=state.now
    audit=audit_ledger(state.cfg,state.network,initial,state.trace)
    net=state.ledger['revenue']-sum(state.ledger[k] for k in COSTS)
    ledger=plain(dict(state.ledger,net=net))
    require(not audit['errors'] and all(abs(dec(v)-dec(audit['ledger'][k]))<dec('1e-7')
        for k,v in ledger.items()),'independent ledger mismatch')
    return dict(net=float(net),ledger=ledger,accepted=sum(o['accepted'] for o in pickups),
        accepted_ids=sorted(o['id'] for o in pickups if o['accepted']),
        trips=sum(e['kind']=='truck_dispatched' for e in state.trace),
        trace=digest(plain(state.trace)), terminal=True,ledger_audited=True,
        pickup_records=plain([dict(id=o['id'],release=o['release'],station=o['station'],
            accepted=o['accepted'],available_at=o['available_at'],completed_at=o['completed_at']) for o in pickups]))


class CapturePolicy(UnifiedVFAPolicy):
    def __init__(self,out,case_id,spec):
        super().__init__(**SETTING,**SEARCH)
        self.out,self.case_id,self.spec=Path(out),case_id,spec
        self.offer_audit=[];self.sites=[];self.environment=None

    def choose(self,known,evaluated):
        from paired_wait_branches import build_branches,qualification
        chosen=super().choose(known,evaluated)
        if not self.selection_log[-1]['feasible_wait_interval_counts']: return chosen
        event=dict(at=float(known.now),case=self.case_id,trucks=[])
        for tid,t in sorted(known.trucks.items()):
            if t['location'] not in known.network['stations'] or t['active_operation'] is not None: continue
            q=qualification(known,chosen,tid,require_just_unloaded=False)
            recent=[e for e in known.trace if e.get('actor')==tid and dec(e['at'])==known.now]
            q['recent_truck_event_kinds']=[e['kind'] for e in recent]
            q['stratum']=('post_unload_primary' if q['just_unloaded'] else
                'post_known_pickup_load' if any(e['kind']=='truck_load_done' for e in recent) else
                'other_departure_ready')
            event['trucks'].append(dict(truck=tid,qualification=q))
            if not q['eligible']: continue
            branches=build_branches(known,chosen,tid,require_just_unloaded=False)
            if set(branches)!={0,2,5,10} or not all(c['feasible'] and c['allowed'] for c in branches.values()):
                event['trucks'][-1]['branch_failure']=plain({h:c.get('reason') for h,c in branches.items()})
                continue
            site_id=f'{self.case_id}_e{len(self.offer_audit):02}_{tid}'
            state=deepcopy(self.environment)
            state._driver_calls.pop(state.now,None)  # snapshot before this one policy invocation
            plans={h:deepcopy(c['plan']) for h,c in branches.items()}
            site=dict(id=site_id,case=self.case_id,spec=self.spec,at=float(known.now),
                truck=tid,station=t['location'],qualification=q,prefix_hash=digest(plain(state.trace)),
                base_plan_hash=canonical_plan_digest(chosen['plan']),
                interventions={str(h):c['intervention'] for h,c in branches.items()})
            save_pickle(self.out/'snapshots'/(site_id+'.pkl.gz'),dict(state=state,plans=plans,site=site))
            write(self.out/'states'/(site_id+'.json'),dict(site=site,known_initial=plain(known.initial),
                ledger=plain(known.ledger),trace_prefix=plain(known.trace),plans=plain(plans),
                live_actions=[dict(at=float(t),actions=plain(rows)) for t,rows in sorted(known._actions.items())],
                physical_queue=plain(known._queue)))
            self.sites.append(site)
        self.offer_audit.append(event)
        return chosen


def capture_case(reference,out):
    spec=reference['world_spec']
    w=make_world(spec['layout_id'],spec['day_id'],spec['total'],spec['share'],spec['pattern'],**WORLD_OPTIONS)
    assert w['input_hash']==reference['input_hash'] and w['i0_hash']==reference['i0_hash']
    case=f"L{spec['layout_id']}_d{spec['day_id']}_N{spec['total']}_{spec['pattern']}"
    state,driver=prepare_online(w['input'],w['configuration'],w['network'],full_day=True)
    state._record_snapshots=False
    policy=CapturePolicy(out,case,spec);driver.policy=policy
    def hook(env):
        policy.environment=env
        driver.on_event(env)
    state.run(decision_hook=hook,stop_at_terminal=True)
    row=final_metrics(state)
    assert row['net']==reference['net'] and row['trace']==reference['trace']
    assert len(policy.offer_audit)==reference['action_search']['feasible_wait_epochs']
    result=dict(case=case,reference=row,sites=policy.sites,offer_audit=policy.offer_audit,
        input_hash=w['input_hash'],reference_reproduced=True)
    write(Path(out)/'capture'/(case+'.json'),result)
    return result


class FixedFirstPolicy(UnifiedVFAPolicy):
    def __init__(self,plan,at):
        super().__init__(**SETTING,**SEARCH)
        self.first_plan,self.first_at=deepcopy(plan),dec(at)
        self.forced=0

    def decide(self,known):
        if not self.forced:
            assert known.now==self.first_at
            assert not known._hidden
            assert not any(e[3] in ('request_reveal','revelation_end') for e in known._queue)
            self.forced=1
            return deepcopy(self.first_plan),dict(at=known.now,selected=self.first_plan['id'],
                decisions=self.first_plan['decisions'],offline_fixed_first_action=True)
        return super().decide(known)


def overlap(state,site):
    t,station=site['at'],site['station']
    rows=[]
    for o in state._hidden.values():
        if station not in o['eligible_stations']: continue
        lag=2*float(distance(o['xy'],state.network['stations'][station]))/state.cfg['drone_speed_km_min']
        lag+=state.cfg['customer_service_per_request']+float(o['weight'])*state.cfg['drone_handling_per_kg']
        ready=float(o['release'])+lag
        rows.append(dict(id=o['id'],release=float(o['release']),idle_drone_ready=ready,
            truck_loaded=ready+float(o['weight'])*state.cfg['truck_handling_per_kg']))
    return dict(optimistic_only=True,formula='release + idle-drone roundtrip + customer service + drone unload + truck load',
        counts={str(h):sum(t<r['truck_loaded']<=t+h for r in rows) for h in [2,5,10]},
        eligible_new_releases_in_window={str(h):sum(t<r['release']<=t+h for r in rows) for h in [2,5,10]},
        idle_ready_in_window={str(h):sum(t<r['idle_drone_ready']<=t+h for r in rows) for h in [2,5,10]},
        eligible_future=len(rows),first_idle_drone_ready=min((r['idle_drone_ready'] for r in rows),default=None),rows=rows)


def evaluate_sample(site,out,replica):
    from paired_wait_suffix import resample_suffix
    out=Path(out)
    path=out/'paired'/(site['id']+f'_r{replica:02}.json')
    if path.exists(): return read(path)
    bundle=load_pickle(out/'snapshots'/(site['id']+'.pkl.gz'))
    if replica<0:
        state=deepcopy(bundle['state'])
        meta=dict(kind='original_realized_suffix',future_orders=plain(list(state._hidden.values())))
    else:
        seed=int(hashlib.sha256(f"wait-pairs-20261128:{site['id']}:{replica}".encode()).hexdigest()[:12],16)
        state,meta=resample_suffix(bundle['state'],seed)
    suffix_hash=digest(plain(state._hidden))
    write(out/'offline_futures'/(site['id']+f'_r{replica:02}.json'),meta)
    before_known=digest(plain(known_clone(state).initial))
    overlap_info=overlap(state,site)
    rows={};start=perf_counter()
    for h in [0,2,5,10]:
        branch=deepcopy(state)
        assert digest(plain(branch._hidden))==suffix_hash
        assert digest(plain(known_clone(branch).initial))==before_known
        branch.start=branch.now;branch._has_run=False
        driver=OnlineCoordinator(FixedFirstPolicy(bundle['plans'][h],branch.now),extend_horizon=True)
        branch.run(decision_hook=driver.on_event,stop_at_terminal=True)
        assert driver.policy.forced==1
        row=final_metrics(branch)
        roads=[e for e in branch.trace[len(state.trace):] if e['kind']=='road_started'
            and e['actor']==site['truck'] and e['source']==site['station']]
        assert roads
        row['actual_first_departure']=float(roads[0]['at'])
        row['actual_first_departure_delay']=float(roads[0]['at'])-site['at']
        if h==0: assert abs(row['actual_first_departure_delay'])<1e-8
        row['continuation_search']=driver.policy.search_metrics()
        row['suffix_hash']=suffix_hash
        rows[str(h)]=row
    if replica<0:
        reference=read(out/'capture'/(site['case']+'.json'))['reference']
        assert rows['0']['net']==reference['net'] and rows['0']['trace']==reference['trace']
    comparisons={}
    base=rows['0']
    for h in [2,5,10]:
        alt=rows[str(h)]
        delta={k:alt['ledger'][k]-base['ledger'][k] for k in base['ledger']}
        gain=alt['net']-base['net']
        assert abs(gain-(delta['revenue']-sum(delta[k] for k in COSTS)))<1e-7
        comparisons[str(h)]=dict(net_gain=gain,accepted_delta=alt['accepted']-base['accepted'],
            added_accepts=sorted(set(alt['accepted_ids'])-set(base['accepted_ids'])),
            lost_accepts=sorted(set(base['accepted_ids'])-set(alt['accepted_ids'])),
            trips_saved=base['trips']-alt['trips'],extra_time_cost=delta['truck_time'],
            ledger_delta=delta,actual_departure_delay=alt['actual_first_departure_delay'])
    result=dict(site=site,replica=replica,suffix_hash=suffix_hash,online_known_hash=before_known,
        future_count=len(state._hidden),overlap=overlap_info,branches=rows,comparisons=comparisons,
        wall_seconds=perf_counter()-start)
    write(path,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['capture','evaluate','all'],default='all')
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--limit-sites',type=int)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.stage in ('capture','all'):
        candidate=read(PRIOR/'training/none/results.json')[0]
        references=next(c['validation'] for c in candidate['curve'] if c['iteration']==candidate['iteration'])
        write(out/'protocol.json',dict(source='none selected validation trajectories',samples=args.samples,
            interventions=[0,2,5,10],continuation='Unified none, no exploration',search=SEARCH,
            setting=SETTING,independent_test_opened=False,source_worlds=[r['world_spec'] for r in references],
            sampling='8 independent conditional future suffixes per state; common across four branches, plus original suffix audit',
            information_boundary='Full suffixes saved offline only; every online call uses known_clone',
            timing_rule='First review may be interrupted by legitimate events, using same continuation policy',
            strata='Primary: just unloaded (15 states); supplement: other departure-ready states. No reserved load is abandoned.',
            expected_offer_events=sum(r['action_search']['feasible_wait_epochs'] for r in references)))
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results=[]
            for future in as_completed([pool.submit(capture_case,r,str(out)) for r in references]):
                row=future.result();results.append(row)
                print(json.dumps(dict(stage='capture',case=row['case'],offers=len(row['offer_audit']),sites=len(row['sites']))),flush=True)
        sites=[s for r in sorted(results,key=lambda r:r['case']) for s in r['sites']]
        write(out/'sites.json',sites)
        assert sum(len(r['offer_audit']) for r in results)==32
    if args.stage in ('evaluate','all'):
        protocol=read(out/'protocol.json')
        assert protocol['samples']==args.samples,'sample count must match predeclared protocol'
        protocol['evaluation_workers']=args.jobs
        write(out/'protocol.json',protocol)
        sites=read(out/'sites.json')
        if args.limit_sites: sites=sites[:args.limit_sites]
        tasks=[(site,replica) for site in sites for replica in [-1]+list(range(args.samples))]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            jobs=[pool.submit(evaluate_sample,s,str(out),r) for s,r in tasks]
            for i,f in enumerate(as_completed(jobs),1):
                r=f.result()
                print(json.dumps(dict(done=i,total=len(jobs),site=r['site']['id'],replica=r['replica'],
                    gains={h:d['net_gain'] for h,d in r['comparisons'].items()})),flush=True)


if __name__=='__main__': main()
