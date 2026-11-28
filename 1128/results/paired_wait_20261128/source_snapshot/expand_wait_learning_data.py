"""Expand only the development/training bank to all pure opportunities on each frozen path."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib

from run_wait_learning_data import OUT, PARENT, PairedPolicy, digest, read, write, cases, execute, inspection, run
from wait_increment_model import wait_features
from run_multiscale import ROOT

OLD = ROOT/'tmp/wait_counterfactual_v1'


def historic(case):
    return case[0]<500


def case_name(case):
    return '_'.join(map(str,case))


class CollectAll(PairedPolicy):
    def __init__(self,model):
        super().__init__(model,None)
        self.x_by_time={}

    def pick(self,known,ranked,theta):
        count=len(self.sites)
        plan,record=super().pick(known,ranked,theta)
        if len(self.sites)>count:
            s=self.sites[-1]
            frozen=next(c for _,_,c in ranked if c['plan'] is plan)
            wait=next(c for _,_,c in ranked if digest(c['plan'])==s['wait'])
            self.x_by_time[s['time']]=wait_features(known,frozen,wait,s['truck'],s['station'])
        return plan,record


def all_sites(case):
    path=OUT/'expanded_sources'/(case_name(case)+'.json')
    if not path.exists():
        w=read((PARENT if historic(case) else OUT)/'worlds'/(digest(case)+'.json'))
        ctl=CollectAll(read(PARENT/'freeze.json')['model'])
        r=run(w,28,policy_override=ctl,metrics=inspection)
        base=read((OLD if historic(case) else OUT)/'runs'/(case_name(case)+'_depart.json'))
        assert all(r[k]==base[k] for k in ('net','trace','plans','accepted','ledger'))
        assert ctl.pure==base['eligible_epochs'] and ctl.chosen_sites==base['selected']
        assert (base['input_hash'],base['i0_hash'])==(w['input_hash'],w['i0_hash'])
        if historic(case):
            assert ctl.sites==base['sites']
        else:
            assert all(x==ctl.x_by_time[t] for t,x in base['x_by_time'].items())
        assert len(ctl.sites)==len(ctl.x_by_time)
        write(path,dict(case=case,sites=ctl.sites,x=ctl.x_by_time))
    return read(path)


def row(case,site,x):
    area=OLD if historic(case) else OUT
    base=read(area/'runs'/(case_name(case)+'_depart.json'))
    if historic(case):
        alt=read(area/'runs'/(case_name(case)+'_wait_'+digest(site)[:16]+'.json'))
    else:
        alt=execute(case,site)
        assert alt['x_by_time'][site['time']]==x
    assert alt['target']==site
    assert (base['input_hash'],base['i0_hash'])==(alt['input_hash'],alt['i0_hash'])
    y=alt['net']-base['net']
    delta={k:alt['ledger'][k]-base['ledger'][k] for k in base['ledger']}
    assert abs(y-(delta['revenue']-sum(v for k,v in delta.items() if k not in ('revenue','net'))))<1e-7
    observed=alt['operation_metrics']['physical'][site['time']]
    assert observed['actual_departure_delay']>0
    return dict(case=case,time=site['time'],truck=site['truck'],station=site['station'],x=x,
                known_plan_delta=site['known_plan_delta'],y=y,depart_net=base['net'],wait_net=alt['net'],
                accepted_delta=alt['accepted']-base['accepted'],delta_ledger=delta,
                input_hash=alt['input_hash'],i0_hash=alt['i0_hash'],prefix=site['prefix'],
                prefix_decisions=site['prefix_decisions'],wait_plan=site['wait'],depart_plan=site['depart'],
                source='previous_wait_diagnostic' if historic(case) else 'current_train_all',
                **observed)


def main():
    from run_high_demand_screen import cases as original_cases
    groups=[original_cases('confirmation'),cases('train')]
    plan=dict(groups=groups,gate='Expand after the original 16 training labels contained no positive gains',
              data='Every pure waiting opportunity on the frozen paths, development and train only',
              code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              frozen=digest(read(PARENT/'freeze.json')['model']))
    pp=OUT/'expanded_data_protocol.json'
    if pp.exists():assert digest(read(pp))==digest(plan)
    write(pp,plan)
    combined=[]
    with ProcessPoolExecutor(max_workers=10) as pool:
        tasks={pool.submit(all_sites,c):c for c in sum(groups,[])}
        jobs=[[],[]]
        for f in as_completed(tasks):
            case=tasks[f]
            ref=f.result()
            i=0 if historic(case) else 1
            jobs[i].extend(pool.submit(row,case,s,ref['x'][s['time']]) for s in ref['sites'])
            print('source',case,len(ref['sites']),flush=True)
        for i, pending in enumerate(jobs):
            rows=[f.result() for f in as_completed(pending)]
            if i==0:
                assert len(rows)==37
                write(OUT/'expanded_historical_train.json',sorted(rows,key=lambda r:(r['case'],float(r['time']))))
            combined.extend(rows)
    bank={(tuple(r['case']),r['time']):r for r in combined}
    assert len(bank)==len(combined)
    for old in read(OUT/'train.json'):
        new=bank[tuple(old['case']),old['time']]
        assert all(old[k]==new[k] for k in ('x','y','prefix','input_hash','i0_hash','wait_plan','depart_plan'))
    ordered=sorted(combined,key=lambda r:(r['case'],float(r['time'])))
    write(OUT/'expanded_train.json',ordered)
    print('COMPLETE',len(ordered),sum(r['y']>1e-7 for r in ordered),flush=True)


if __name__=='__main__':
    main()
