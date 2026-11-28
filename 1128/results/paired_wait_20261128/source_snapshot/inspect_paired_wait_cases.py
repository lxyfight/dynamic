"""Reproduce complete traces for selected positive/negative offline wait pairs."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

from e1_sim.online import OnlineCoordinator
from e1_sim.quantities import plain
from paired_wait_suffix import resample_suffix
from run_paired_wait_diagnostic import (DEFAULT_OUT,FixedFirstPolicy,final_metrics,
                                        load_pickle,read,write)


def compact_decision(row):
    """Keep selected decisions; candidate pools are not needed for trace audit."""
    result={k:plain(row[k]) for k in ('at','selected','reason','decisions','selected_J',
        'policy_called','offline_fixed_first_action') if k in row}
    joint=row.get('joint_candidate_audit',{})
    if joint:
        result['chosen']={k:plain(joint[k]) for k in ('J','W','features','plan_hash',
            'chosen_wait_intervals') if k in joint}
    return result


def inspect(path,output):
    paired=read(path)
    site,replica=paired['site'],paired['replica']
    root=path.parent.parent
    bundle=load_pickle(root/'snapshots'/(site['id']+'.pkl.gz'))
    if replica<0:
        state=deepcopy(bundle['state'])
    else:
        seed=int(hashlib.sha256(f"wait-pairs-20261128:{site['id']}:{replica}".encode()).hexdigest()[:12],16)
        state,_=resample_suffix(bundle['state'],seed)
    results={}
    for h in (0,2,5,10):
        branch=deepcopy(state)
        branch.start=branch.now;branch._has_run=False
        driver=OnlineCoordinator(FixedFirstPolicy(bundle['plans'][h],branch.now),extend_horizon=True)
        branch.run(decision_hook=driver.on_event,stop_at_terminal=True)
        metrics=final_metrics(branch)
        expected=paired['branches'][str(h)]
        assert all(metrics[k]==expected[k] for k in ('net','trace','ledger','accepted','accepted_ids','trips'))
        loads=[plain(e) for e in branch.trace if e['kind']=='truck_load_started']
        roads=[plain(e) for e in branch.trace if e['kind']=='road_started']
        requests=[]
        for oid,o in branch.orders.items():
            if o['kind']!='P':continue
            requests.append(dict(order=plain(o),
                admission_events=[compact_decision(d) for d in driver.decisions if oid in d.get('decisions',{})],
                truck_loads=[e for e in loads if oid in e.get('orders',[])],
                recovery_at_intervention_station=o['station']==site['station']))
        results[str(h)]=dict(metrics=metrics,trace=plain(branch.trace),
            orders=plain(list(branch.orders.values())),pickup_details=requests,
            truck_road_starts=roads,truck_station_departures=[e for e in roads if e['source'] in branch.network['stations']],
            decision_records=[compact_decision(d) for d in driver.decisions],exact_original_pair_reproduced=True)
        print('reproduced',site['id'],replica,h,metrics['net'],flush=True)
    differences={}
    for h in ('2','5','10'):
        base=results['0']['metrics'];alt=results[h]['metrics']
        extra=set(alt['accepted_ids'])-set(base['accepted_ids'])
        lost=set(base['accepted_ids'])-set(alt['accepted_ids'])
        differences[h]=dict(net_gain=alt['net']-base['net'],
            ledger_delta={k:alt['ledger'][k]-base['ledger'][k] for k in base['ledger']},
            added_pickups=[r for r in results[h]['pickup_details'] if r['order']['id'] in extra],
            lost_pickups=[r for r in results['0']['pickup_details'] if r['order']['id'] in lost],
            trips_saved=base['trips']-alt['trips'])
    value=dict(site=site,replica=replica,source_pair=str(path),
        selection='diagnostic example selected after outcomes; not an independent performance estimate',
        original_comparisons=paired['comparisons'],branches=results,differences=differences)
    output.mkdir(parents=True,exist_ok=True)
    write(output/path.name,value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--site',default='L112881_d0_N40_dual_e00_I0E00001')
    a=p.parse_args();root=a.output.resolve()
    paths=list((root/'paired').glob(a.site+'_r*.json'))
    sampled=[(path,read(path)) for path in paths if read(path)['replica']>=0]
    assert len(sampled)==8,'select after all eight predeclared samples finish'
    positive=max(sampled,key=lambda x:x[1]['comparisons']['2']['net_gain'])
    negative=min(sampled,key=lambda x:(x[1]['comparisons']['10']['net_gain'],x[1]['replica']))
    selected=[]
    for tag,(path,value) in [('largest_wait2_gain',positive),('negative_example',negative)]:
        inspect(path,root/'audit/details')
        selected.append(dict(reason=tag,file=path.name,replica=value['replica'],
            comparisons=value['comparisons']))
    write(root/'audit/details/selection.json',dict(site=a.site,selected=selected,
        caveat='Post-hoc mechanism examples; use all predeclared samples for means.'))


if __name__=='__main__':main()
