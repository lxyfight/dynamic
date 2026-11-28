"""Independent structural/arithmetic audit of three completed paired banks.

Does not rerun optimization. Each trajectory's physical ledger audit flag is
checked, and saved ledger arithmetic, prefix/suffix identities and original
continuation replication are independently verified here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from e1_sim.initializer import digest
from e1_sim.kernel import COSTS
from e1_sim.online import known_clone
from e1_sim.quantities import plain,dec
from e1_sim.runner import DEFAULT_SPEC
from run_paired_wait_diagnostic import load_pickle,read,write

SPLITS=('train','tune','dev')
RESERVED={112891,112892,112893,112894}
EXPECTED_CASES={
    'train':{(l,d,80,p) for l in (112871,112872) for d in (1,2) for p in ('uniform','dual')},
    'tune':{(l,1,80,p) for l in (112901,112902) for p in ('uniform','dual')},
    'dev':{(l,2 if l%2 else 3,80,p) for l in range(112903,112911) for p in ('uniform','dual')},
}
SEARCH_SUMS=('epochs','truncated_epochs','base_search_truncated_epochs',
    'enhancement_truncated_epochs','service_search_truncated_epochs',
    'wait_search_truncated_epochs','forced_reject_requests',
    'service_patterns_attempted','service_patterns_accepted','wait_patterns_attempted',
    'wait_patterns_accepted','feasible_wait_epochs','decision_seconds')


class Checks:
    def __init__(self):self.count=0;self.failures=[]
    def check(self,condition,label,context=''):
        self.count+=1
        if not condition:self.failures.append(dict(check=label,context=context))


def close(a,b):return abs(dec(a)-dec(b))<=dec('0.0000001')


def audit_pair(row,metadata,site,replica,known_hash,reference,checks):
    context=site['id']+':'+str(replica)
    checks.check(row['site']==site and row['replica']==replica,'pair_identity',context)
    checks.check(row['online_known_hash']==known_hash,'same_predecision_known_state',context)
    orders=metadata['future_orders']
    hidden={o['id']:dict(o,release=float(o['release']),deadline=float(o['deadline'])) for o in orders}
    checks.check(len(hidden)==len(orders)==row['future_count'],'future_count_and_unique_ids',context)
    checks.check(digest(hidden)==row['suffix_hash'],'saved_future_reproduces_suffix_hash',context)
    checks.check(all(float(o['release'])>site['at'] for o in orders),'future_strictly_after_state',context)
    checks.check(all(any(close(float(o['deadline'])-float(o['release']),lag)
        for lag in (240,300,360)) for o in orders),
        'deadline_support',context)
    checks.check(set(row['branches'])=={'0','2','5','10'},'four_branches',context)
    checks.check(set(row['comparisons'])=={'2','5','10'},'three_differences',context)
    for h,branch in row['branches'].items():
        c=context+':h'+h
        checks.check(branch['terminal'] is True and branch['ledger_audited'] is True,'physical_audit_recorded',c)
        checks.check(branch['suffix_hash']==row['suffix_hash'],'same_suffix_four_branches',c)
        ledger=branch['ledger']
        computed=dec(ledger['revenue'])-sum(dec(ledger[k]) for k in COSTS)
        checks.check(close(computed,branch['net']) and close(computed,ledger['net']),'ledger_net_arithmetic',c)
        checks.check(branch['accepted']==len(set(branch['accepted_ids'])),'accept_count_and_ids',c)
        checks.check(branch['continuation_search']['solver']=='gurobi','same_gurobi_continuation',c)
        checks.check(branch['actual_first_departure_delay']>=-1e-8,'departure_not_before_intervention',c)
    base=row['branches']['0']
    checks.check(close(base['actual_first_departure_delay'],0),'zero_branch_departs_immediately',context)
    if replica==-1:
        checks.check(all(base[k]==reference[k] for k in ('net','trace','ledger','accepted','accepted_ids','trips')),
                     'original_future_zero_branch_exact_replication',context)
    else:
        expected_seed=int(hashlib.sha256(f"wait-pairs-20261128:{site['id']}:{replica}".encode()).hexdigest()[:12],16)
        checks.check(metadata['seed']==expected_seed and metadata['online_information_unchanged'] is True
            and metadata['original_suffix_used'] is False,'conditional_seed_and_information_boundary',context)
    for h,difference in row['comparisons'].items():
        alt=row['branches'][h];c=context+':h'+h
        expected={k:dec(alt['ledger'][k])-dec(base['ledger'][k]) for k in base['ledger']}
        checks.check(all(close(v,difference['ledger_delta'][k]) for k,v in expected.items()),'ledger_difference',c)
        checks.check(close(difference['net_gain'],dec(alt['net'])-dec(base['net'])),'profit_difference',c)
        checks.check(close(difference['extra_time_cost'],expected['truck_time']),'extra_time_cost',c)
        checks.check(difference['trips_saved']==base['trips']-alt['trips'],'trips_saved',c)
        checks.check(difference['accepted_delta']==alt['accepted']-base['accepted'],'accepted_difference',c)
        checks.check(set(difference['added_accepts'])==set(alt['accepted_ids'])-set(base['accepted_ids'])
            and set(difference['lost_accepts'])==set(base['accepted_ids'])-set(alt['accepted_ids']),
            'added_and_lost_orders',c)


def search_totals():
    return dict(totals={key:0 for key in SEARCH_SUMS},chosen_wait_intervals=Counter(),
        feasible_wait_interval_epochs=Counter(),branch_count=0)


def add_search(target,search):
    target['branch_count']+=1
    for key in SEARCH_SUMS:target['totals'][key]+=search[key]
    target['chosen_wait_intervals'].update(search['chosen_wait_interval_counts'])
    target['feasible_wait_interval_epochs'].update(search['feasible_wait_interval_epochs'])


def audit_campaign(root):
    root=Path(root)
    missing=[s for s in SPLITS if not (root/s/'evaluation_complete.json').is_file()]
    if missing:
        return dict(successful=False,status='incomplete',missing_completed_splits=missing,
            check_count=0,scope='No partial-result success audit was attempted.')
    checks=Checks();report={};layouts={};all_site_ids=set();known_hash_splits={};runtime=None;common=None
    for split in SPLITS:
        directory=root/split;protocol=read(directory/'protocol.json');done=read(directory/'evaluation_complete.json')
        sites=read(directory/'sites.json');cases=protocol['cases'];samples=protocol['samples']
        layouts[split]={s['layout_id'] for s in cases}
        checks.check(not layouts[split]&RESERVED,'no_reserved_final_test_layout',split)
        checks.check(protocol['independent_test_opened'] is False and done['independent_test_opened'] is False,'test_unopened',split)
        identities=[(s['layout_id'],s['day_id'],s['total'],s['pattern']) for s in cases]
        checks.check(len(identities)==len(set(identities)),'unique_cases',split)
        checks.check(set(identities)==EXPECTED_CASES[split] and all(s['share']==.6 for s in cases),
                     'exact_predeclared_bank',split)
        current_common={k:protocol[k] for k in ('search','setting','world_options','seed','source_policy','continuation_policy')}
        if common is None:common=current_common
        checks.check(current_common==common,'common_runtime_search_and_policy',split)
        checks.check(done['primary_states']==len(sites) and done['conditional_replicas_per_state']==samples
            and done['original_audit_replicas_per_state']==1 and done['paired_samples']==len(sites)*(samples+1)
            and done['full_day_branches']==4*len(sites)*(samples+1),'completion_counts',split)
        checks.check(all(s['qualification']['eligible'] and s['qualification']['just_unloaded']
            and s['qualification']['stratum']=='post_unload_primary' for s in sites),'primary_only',split)
        if runtime is None:runtime=protocol['source_hashes']
        checks.check(runtime==protocol['source_hashes'],'same_runtime_between_splits',split)
        for name,expected in protocol['source_hashes'].items():
            path=DEFAULT_SPEC if name=='data/fixtures/e1_cases.json' else Path(__file__).resolve().parent/name
            checks.check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,'runtime_source_hash',split+':'+name)
        capture_files=list((directory/'capture').glob('*.json'))
        checks.check(len(capture_files)==len(cases),'capture_file_count',split)
        captures={read(p)['case']:read(p) for p in capture_files}
        captured_ids={s['id'] for c in captures.values() for s in c['sites']}
        checks.check(captured_ids=={s['id'] for s in sites},'sites_match_capture_records',split)
        checks.check(len(sites)==len({s['id'] for s in sites}) and not all_site_ids&captured_ids,'unique_cross_split_site_ids',split)
        all_site_ids.update(captured_ids)
        aggregate={kind:{str(h):search_totals() for h in (0,2,5,10)} for kind in ('conditional','original_suffix_audit')}
        expected_files=set();prefix_hashes=set();future_keys=set()
        for site in sites:
            checks.check((site['spec']['layout_id'],site['spec']['day_id'],site['spec']['total'],site['spec']['pattern']) in identities,'state_from_declared_case',site['id'])
            bundle=load_pickle(directory/'snapshots'/(site['id']+'.pkl.gz'))
            state=bundle['state'];known=known_clone(state);known_hash=digest(plain(known.initial))
            checks.check(bundle['site']==site and digest(plain(state.trace))==site['prefix_hash'],'snapshot_site_prefix',site['id'])
            checks.check(not known._hidden and not any(e[3] in ('request_reveal','revelation_end') for e in known._queue),'sanitized_online_state',site['id'])
            checks.check(any(e['kind']=='truck_unload_done' and e.get('actor')==site['truck']
                and e.get('station')==site['station'] and dec(e['at'])==state.now for e in state.trace),'real_unload_event_at_snapshot',site['id'])
            checks.check(known_hash not in known_hash_splits or known_hash_splits[known_hash]==split,'no_exact_public_state_reuse_between_splits',site['id'])
            known_hash_splits[known_hash]=split;prefix_hashes.add(known_hash)
            for replica in [-1]+list(range(samples)):
                name=site['id']+f'_r{replica:02}.json';expected_files.add(name)
                path=directory/'paired'/name;future_path=directory/'offline_futures'/name
                checks.check(path.is_file() and future_path.is_file(),'paired_and_offline_sample_exist',name)
                if not path.is_file() or not future_path.is_file():continue
                row=read(path);metadata=read(future_path)
                checks.check(not set(o['id'] for o in metadata['future_orders'])&set(state.orders),'future_ids_exclude_revealed_prefix',name)
                audit_pair(row,metadata,site,replica,known_hash,captures[site['case']]['reference'],checks)
                if replica>=0:
                    checks.check((site['id'],metadata['seed']) not in future_keys,'unique_conditional_replica_seed',name)
                    future_keys.add((site['id'],metadata['seed']))
                kind='original_suffix_audit' if replica<0 else 'conditional'
                for h,branch in row['branches'].items():add_search(aggregate[kind][h],branch['continuation_search'])
        checks.check(expected_files=={p.name for p in (directory/'paired').glob('*.json')},'exact_sample_files_no_missing_or_extra',split)
        checks.check(expected_files=={p.name for p in (directory/'offline_futures').glob('*.json')},'exact_offline_future_files',split)
        report[split]=dict(cases=len(cases),layouts=len(layouts[split]),primary_states=len(sites),
            unique_public_state_hashes=len(prefix_hashes),samples_per_state=samples,
            conditional_paired_samples=len(sites)*samples,original_suffix_checks=len(sites),
            full_day_branches=4*len(sites)*(samples+1),search_by_sample_kind_and_initial_interval=aggregate,
            time_limit=dict(configured=False,reported_hits=None,
                explanation='No solver wall-time limit configured/reported; truncation is candidate-count budget based.'),
            search_count_note='Counts sum simulated continuation events across replicas/branches; truncation sources overlap and must not be summed as unique events.',
            independence='Layouts differ between splits; states within a world and paired spatial variants are dependent.')
    for i,left in enumerate(SPLITS):
        for right in SPLITS[i+1:]:checks.check(not layouts[left]&layouts[right],'disjoint_layouts',left+':'+right)
    return dict(successful=not checks.failures,status='passed' if not checks.failures else 'failed',
        check_count=checks.count,failed_checks=checks.failures,splits=report,
        limitation='Checks stored physical-audit/terminal flags and independently recomputes accounting arithmetic; does not rerun each optimization trajectory.',
        independent_test_opened=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('../../results/paired_chain_20261128'))
    args=p.parse_args();root=args.output.resolve()
    try:result=audit_campaign(root)
    except Exception as exc:result=dict(successful=False,status='failed',error=type(exc).__name__+': '+str(exc))
    write(root/'audit/paired_campaign_integrity.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='splits'},ensure_ascii=False),flush=True)
    if not result['successful']:raise SystemExit(1)


if __name__=='__main__':main()
