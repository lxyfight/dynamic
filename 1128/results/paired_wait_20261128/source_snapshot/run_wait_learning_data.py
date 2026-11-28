"""Shared-realization data bank for a five-minute station waiting decision."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
from time import monotonic, sleep

import run_high_demand_screen as factory
from run_wait_counterfactual import PairedPolicy, pure_pairs, inspection, run
from run_multiscale import ROOT, digest, read, write

OUT = ROOT/'tmp/wait_learning_v1'
PARENT = ROOT/'tmp/high_demand_screen_v1'
SPLITS = dict(train=range(520,524), validation=range(524,526), test=range(526,530))


def cases(stage):
    return [(layout,100 if layout%2==0 else 150,shape)
            for layout in SPLITS[stage] for shape in ('uniform','dual')]


def prepare(case):
    factory.OUT = OUT
    w = factory.world_for(case)
    assert digest(w['input']) == w['input_hash']
    return case


class InspectPair(PairedPolicy):
    def __init__(self, model, target):
        super().__init__(model,target)
        self.x_by_time = {}

    def pick(self,known,ranked,theta):
        before = len(self.sites)
        plan, record = super().pick(known,ranked,theta)
        if before == len(self.sites):
            return plan,record
        site = self.sites[-1]
        if self.target is None and site not in self.chosen_sites:
            return plan,record
        from wait_increment_model import wait_features
        dep = next(c for _,_,c in ranked if digest(c['plan']) == site['depart'])
        alt = next(c for _,_,c in ranked if digest(c['plan']) == site['wait'])
        self.x_by_time[site['time']] = wait_features(known,dep,alt,site['truck'],site['station'])
        return plan,record


def execute(case,target=None):
    world_path = OUT/'worlds'/(digest(case)+'.json')
    limit = monotonic()+600
    while not world_path.exists():
        if monotonic()>limit:
            raise RuntimeError(f'Missing prepared world {case}')
        sleep(1)
    w = read(world_path)
    model = read(PARENT/'freeze.json')['model']
    tag = 'depart' if target is None else 'wait_'+digest(target)[:16]
    path = OUT/'runs'/('_'.join(map(str,(*case,tag)))+'.json')
    if not path.exists():
        pol = InspectPair(model,target)
        row = run(w,28,policy_override=pol,metrics=inspection)
        assert pol.forced == (0 if target is None else 1)
        row.update(case=case,input_hash=w['input_hash'],i0_hash=w['i0_hash'],
                   model_hash=digest(model),target=target,selected=pol.chosen_sites,
                   offer_epochs=pol.offers,eligible_epochs=pol.pure,x_by_time=pol.x_by_time)
        if target is None:
            assert pol.offers == row['operation_metrics']['decisions_with_new_wait']
        write(path,row)
    row = read(path)
    assert tuple(row['case'])==case and row['target']==target
    assert (row['input_hash'],row['i0_hash'],row['model_hash'])==(w['input_hash'],w['i0_hash'],digest(model))
    ledger=row['ledger']
    assert abs(row['net']-(ledger['revenue']-sum(v for k,v in ledger.items() if k not in ('revenue','net'))))<1e-7
    return row


def bank(pool,stage):
    specs=cases(stage)
    jobs={pool.submit(execute,c):c for c in specs}
    baselines={}
    alternatives=[]
    for future in as_completed(jobs):
        case=jobs[future]
        base=future.result()
        baselines[case]=base
        for site in base['selected']:
            alternatives.append(pool.submit(execute,case,site))
        print(stage,'depart',case,len(base['selected']),flush=True)
    rows=[]
    for future in as_completed(alternatives):
        r=future.result()
        case=tuple(r['case'])
        b=baselines[case]
        s=r['target']
        t=s['time']
        assert (r['input_hash'],r['i0_hash'])==(b['input_hash'],b['i0_hash']) and s in b['selected']
        assert r['x_by_time'][t]==b['x_by_time'][t]
        alt=r['operation_metrics']['physical'][t]
        ref=b['operation_metrics']['physical'][t]
        assert ref['actual_departure_delay']==0 and alt['actual_departure_delay']>0
        assert alt['interval_new_pickups']==ref['interval_new_pickups']
        assert alt['interval_new_station_eligible']==ref['interval_new_station_eligible']
        delta={k:r['ledger'][k]-b['ledger'][k] for k in b['ledger']}
        y=r['net']-b['net']
        assert abs(y-(delta['revenue']-sum(v for k,v in delta.items() if k not in ('revenue','net'))))<1e-7
        rows.append(dict(case=list(case),time=t,truck=s['truck'],station=s['station'],x=r['x_by_time'][t],
                         known_plan_delta=s['known_plan_delta'],y=y,wait_net=r['net'],depart_net=b['net'],
                         accepted_delta=r['accepted']-b['accepted'],delta_ledger=delta,
                         input_hash=r['input_hash'],i0_hash=r['i0_hash'],prefix=s['prefix'],
                         prefix_decisions=s['prefix_decisions'],wait_plan=s['wait'],depart_plan=s['depart'],**alt))
        print(stage,'pair',case,t,flush=True)
    assert len(rows)==sum(len(b['selected']) for b in baselines.values())
    assert len({(tuple(r['case']),r['time']) for r in rows})==len(rows)
    rows.sort(key=lambda r:(r['case'],float(r['time'])))
    write(OUT/(stage+'.json'),rows)
    print('COMPLETE',stage,len(rows),flush=True)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('phase',choices=['worlds',*SPLITS,'all'])
    cli.add_argument('--workers',type=int,default=8)
    args=cli.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.phase=='worlds':
            futures=[pool.submit(prepare,sp) for stage in SPLITS for sp in cases(stage)]
            for f in as_completed(futures):
                print('world',f.result(),flush=True)
            return
        src=Path(__file__).parent
        source_files=[Path(__file__),src/'wait_increment_model.py',src/'run_wait_counterfactual.py']
        protocol=dict(cases={stage:cases(stage) for stage in SPLITS}, selection='first pure event and first distinct pure event at/after 300 minutes',
                      horizon=5, response='paired realized full-day net, wait minus depart',
                      model=digest(read(PARENT/'freeze.json')['model']),
                      source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
        path=OUT/'data_protocol.json'
        if path.exists():
            assert digest(read(path))==digest(protocol),'data protocol changed'
        write(path,protocol)
        for stage in (list(SPLITS) if args.phase=='all' else [args.phase]):
            bank(pool,stage)


if __name__=='__main__':
    main()
