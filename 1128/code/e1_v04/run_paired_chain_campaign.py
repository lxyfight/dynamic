"""Frozen train/tune/new-development banks for conditional paired wait returns.

Only freshly unloaded departure-ready states enter the paired sample bank.
No legacy validation or reserved final-test layout is evaluated here.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import hashlib
from pathlib import Path
import json

from e1_sim.initializer import digest
from e1_sim.online import prepare_online
from e1_sim.quantities import dec,plain
from e1_sim.runner import DEFAULT_SPEC
from paired_wait_branches import build_branches,qualification
from run_feature_screen import SEARCH,SETTING,WORLD_OPTIONS,write
from run_paired_wait_diagnostic import (CapturePolicy,evaluate_sample,final_metrics,
                                      read,save_pickle)
from unified_vfa_policy import UnifiedVFAPolicy,canonical_plan_digest
from uniform_wait_worlds import make_world

SPLITS=dict(train=dict(layouts=[112871,112872],days=[1,2],samples=16),
    tune=dict(layouts=[112901,112902],days=[1],samples=24),
    dev=dict(layouts=list(range(112903,112911)),days=[2,3],samples=32,
             day_assignment='odd_layout_day2_even_layout_day3'))
RESERVED_TEST={112891,112892,112893,112894}
DEFAULT_OUTPUT=Path('../../results/paired_chain_campaign_20261128')


def specs(split):
    design=SPLITS[split]
    if split=='dev':
        return [dict(layout_id=l,day_id=2 if l%2 else 3,total=80,share=.6,pattern=p)
                for l in design['layouts'] for p in ('uniform','dual')]
    return [dict(layout_id=l,day_id=d,total=80,share=.6,pattern=p)
        for l in design['layouts'] for d in design['days'] for p in ('uniform','dual')]


def case_id(spec,seed):
    return 's{}_L{}_d{}_N{}_{}'.format(seed,spec['layout_id'],spec['day_id'],spec['total'],spec['pattern'])


def source_hashes():
    root=Path(__file__).resolve().parent
    names=['run_paired_chain_campaign.py','run_paired_wait_diagnostic.py',
        'paired_wait_branches.py','paired_wait_suffix.py','run_feature_screen.py',
        'uniform_wait_worlds.py','unified_vfa_policy.py','unified_vfa_features.py',
        'joint_action_mip.py','service_action_mip.py','run_global_demand_pilot.py',
        'run_multiscale.py']
    files=[root/n for n in names]+sorted((root/'e1_sim').glob('*.py'))
    result={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result['data/fixtures/e1_cases.json']=hashlib.sha256(DEFAULT_SPEC.read_bytes()).hexdigest()
    return result


def protocol_for(split,samples,seed):
    if samples<1:raise ValueError('positive conditional samples required')
    cases=specs(split)
    assert not any(s['layout_id'] in RESERVED_TEST for s in cases)
    return dict(version='paired-chain-primary-campaign-v1',split=split,samples=samples,seed=seed,
        cases=cases,split_design=SPLITS,interventions=[0,2,5,10],
        source_policy='none, no exploration',continuation_policy='same none, no exploration',
        state_filter='post_unload_primary only; genuine immediate departure; no loading reservation',
        conditional_sampling='one sampled suffix per state/replica shared by all four branches',
        seed_rule='evaluate_sample SHA256 of site ID and replica; seed enters case/site ID',
        original_suffix_replica=-1,original_suffix_in_training_labels=False,
        search=SEARCH,setting=SETTING,world_options=WORLD_OPTIONS,source_hashes=source_hashes(),
        reserved_final_test_layouts=sorted(RESERVED_TEST),independent_test_opened=False,
        scope='train/tune/new development, not final test',
        dependence='States within a world are dependent; spatial variants share random streams.')


def lock_protocol(output,protocol,*,create):
    path=Path(output)/'protocol.json'
    if path.exists():
        if read(path)!=protocol:raise ValueError('protocol/source mismatch; use a new output root')
    elif create:
        write(path,protocol)
    else:raise ValueError('capture stage and its locked protocol must exist before evaluation')


class PrimaryCapturePolicy(CapturePolicy):
    def choose(self,known,evaluated):
        chosen=UnifiedVFAPolicy.choose(self,known,evaluated)
        if not self.selection_log[-1]['feasible_wait_interval_counts']:return chosen
        event=dict(at=float(known.now),case=self.case_id,trucks=[])
        for tid,t in sorted(known.trucks.items()):
            if t['location'] not in known.network['stations'] or t['active_operation'] is not None:continue
            q=qualification(known,chosen,tid,require_just_unloaded=True)
            recent=[e['kind'] for e in known.trace if e.get('actor')==tid and dec(e['at'])==known.now]
            q.update(recent_truck_event_kinds=recent,stratum=('post_unload_primary' if q.get('just_unloaded')
                else 'post_known_pickup_load' if 'truck_load_done' in recent else 'other_departure_ready'))
            audit=dict(truck=tid,qualification=q);event['trucks'].append(audit)
            if not q['eligible']:continue
            try:branches=build_branches(known,chosen,tid,require_just_unloaded=True)
            except ValueError as exc:
                audit['branch_failure']=str(exc)
                continue
            site_id=f'{self.case_id}_e{len(self.offer_audit):02}_{tid}'
            state=deepcopy(self.environment);state._driver_calls.pop(state.now,None)
            plans={h:deepcopy(c['plan']) for h,c in branches.items()}
            site=dict(id=site_id,case=self.case_id,spec=self.spec,at=float(known.now),truck=tid,
                station=t['location'],qualification=q,prefix_hash=digest(plain(state.trace)),
                base_plan_hash=canonical_plan_digest(chosen['plan']),
                interventions={str(h):c['intervention'] for h,c in branches.items()})
            save_pickle(self.out/'snapshots'/(site_id+'.pkl.gz'),dict(state=state,plans=plans,site=site))
            write(self.out/'states'/(site_id+'.json'),dict(site=site,known_initial=plain(known.initial),
                ledger=plain(known.ledger),trace_prefix=plain(known.trace),plans=plain(plans),
                live_actions=[dict(at=float(at),actions=plain(rows)) for at,rows in sorted(known._actions.items())],
                physical_queue=plain(known._queue)))
            self.sites.append(site)
        self.offer_audit.append(event)
        return chosen


def capture_case(spec,output,seed):
    out=Path(output);case=case_id(spec,seed);path=out/'capture'/(case+'.json')
    if path.exists():
        row=read(path)
        assert row['spec']==spec and row['case']==case and row['capture_complete']
        for s in row['sites']:
            assert s['qualification']['eligible'] and s['qualification']['stratum']=='post_unload_primary'
            assert (out/'snapshots'/(s['id']+'.pkl.gz')).is_file()
            assert (out/'states'/(s['id']+'.json')).is_file()
        return row
    w=make_world(**spec,**WORLD_OPTIONS)
    state,driver=prepare_online(w['input'],w['configuration'],w['network'],full_day=True)
    state._record_snapshots=False
    policy=PrimaryCapturePolicy(out,case,spec);driver.policy=policy
    def hook(env):policy.environment=env;driver.on_event(env)
    state.run(decision_hook=hook,stop_at_terminal=True)
    reference=final_metrics(state)
    excluded=Counter(reason for e in policy.offer_audit for t in e['trucks']
        for reason in t['qualification']['reasons'])
    failures=[dict(at=e['at'],truck=t['truck'],reason=t['branch_failure'])
        for e in policy.offer_audit for t in e['trucks'] if 'branch_failure' in t]
    row=dict(case=case,spec=spec,reference=reference,sites=policy.sites,offer_audit=policy.offer_audit,
        excluded_reason_counts=dict(excluded),branch_failures=failures,capture_complete=True,
        input_hash=w['input_hash'],i0_hash=w['i0_hash'],source='new none trajectory, not old validation reference')
    write(path,row)
    return row


def evaluate_cached(site,output,replica):
    row=evaluate_sample(site,output,replica)
    assert row['site']==site and row['replica']==replica
    assert set(row['branches'])=={'0','2','5','10'}
    assert all(r['terminal'] and r['ledger_audited'] and r['suffix_hash']==row['suffix_hash']
               for r in row['branches'].values())
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['capture','evaluate'])
    p.add_argument('--split',required=True,choices=list(SPLITS))
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--samples',type=int)
    p.add_argument('--seed',type=int,default=20261128)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    if args.jobs<1:p.error('--jobs must be positive')
    samples=SPLITS[args.split]['samples'] if args.samples is None else args.samples
    out=args.output.resolve()/args.split
    protocol=protocol_for(args.split,samples,args.seed)
    lock_protocol(out,protocol,create=args.stage=='capture')
    if args.stage=='capture':
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            jobs=[pool.submit(capture_case,s,str(out),args.seed) for s in specs(args.split)]
            results=[]
            for f in as_completed(jobs):
                row=f.result();results.append(row)
                print(json.dumps(dict(stage='capture',split=args.split,case=row['case'],
                    offer_events=len(row['offer_audit']),primary_states=len(row['sites']))),flush=True)
        results.sort(key=lambda r:r['case'])
        sites=[s for r in results for s in r['sites']]
        write(out/'sites.json',sites)
        excluded=Counter()
        for r in results:excluded.update(r['excluded_reason_counts'])
        write(out/'capture_summary.json',dict(cases=len(results),primary_states=len(sites),
            offer_events=sum(len(r['offer_audit']) for r in results),excluded_reason_counts=dict(excluded),
            branch_failures=[f for r in results for f in r['branch_failures']],
            per_case=[dict(case=r['case'],primary_states=len(r['sites']),
                offers=len(r['offer_audit']),excluded=r['excluded_reason_counts']) for r in results]))
    else:
        sites=read(out/'sites.json')
        assert all(s['qualification']['stratum']=='post_unload_primary' and s['qualification']['eligible'] for s in sites)
        if not sites:raise ValueError('no qualifying states; inspect exclusions before proceeding')
        tasks=[(s,r) for s in sites for r in [-1]+list(range(samples))]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            jobs=[pool.submit(evaluate_cached,s,str(out),r) for s,r in tasks]
            for i,f in enumerate(as_completed(jobs),1):
                row=f.result()
                print(json.dumps(dict(done=i,total=len(tasks),split=args.split,site=row['site']['id'],
                    replica=row['replica'],gains={h:d['net_gain'] for h,d in row['comparisons'].items()})),flush=True)
        write(out/'evaluation_complete.json',dict(split=args.split,primary_states=len(sites),
            conditional_replicas_per_state=samples,original_audit_replicas_per_state=1,
            paired_samples=len(tasks),full_day_branches=4*len(tasks),independent_test_opened=False))


if __name__=='__main__':main()
