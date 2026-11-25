"""Predeclared high-demand feature screen, paired across policies on full episodes."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
from pathlib import Path
from random import Random
from statistics import mean
from high_demand_features import GROUPS, NAMES, run_case
from high_demand_screen import screen
from run_global_demand_pilot import (make_world, read, write, ROOT, point, release_slots,
                                    BANDS, seed, digest, validate_public)

OUT = ROOT/'tmp/high_demand_screen_v1'
SPLITS = {'fit':[410,411,412,413], 'value_validation':[420,421],
          'policy_validation':[430,431], 'confirmation':[440,441,442,443]}


def cases(phase):
    # Counterbalance N and spatial pattern in each pair of layouts. Within a
    # layout the two patterns have the SAME N, dynamic share, weights and times.
    return [(layout, 100 if layout%2==0 else 150, shape)
            for layout in SPLITS[phase] for shape in ('uniform','dual')]


def world_for(spec):
    path = OUT/'worlds'/(digest(spec)+'.json')
    if path.exists():
        return read(path)
    layout,n,shape = spec
    world = make_world(layout,2,n,.4,shape)
    # Fit the public forecast on independent diverse historical days. The
    # current layout and spatial law are public, evaluated N/realized P are not.
    rng = Random(seed('independent-high-history',world['network']['city_x'],shape))
    counts = Counter()
    for _ in range(160):
        size = rng.choice((75,100,125,150,175))
        share = rng.choice((.2,.4,.5,.6))
        for t in release_slots(rng,round(size*share)):
            city,_ = point(rng,world['network'],shape)
            counts[city,next(lo for lo,hi,_ in BANDS if lo<=t<=hi)] += 1
    raw = world['input']
    law = raw['public_information']
    for city in law['demand_model']['rates_by_city']:
        law['demand_model']['rates_by_city'][city] = [
            dict(start=lo,end=hi,rate=counts[city,lo]/(160*(hi-lo+1))) for lo,hi,_ in BANDS]
    validate_public(law)
    assert len(raw['orders'])==n and sum(o['kind']=='P' for o in raw['orders'])==round(n*.4)
    assert world['configuration']['truck_handling_setup']==0
    world['input_hash'] = digest(raw)
    write(path,world)
    return world


def worker(task):
    spec, label, model, collect = task
    key = digest([spec,label,model,collect])
    path = OUT/'runs'/(key+'.json')
    if path.exists():
        return read(path)
    w = world_for(spec)
    theta = 0 if label=='Myopic' else 28
    row = run_case(w,model=model,teacher=theta,collect=collect,
                   explore_seed=seed('high-bank',*spec) if collect else None)
    row.update(layout=spec[0],case=spec,policy=label)
    write(path,row)
    return row


def batch(pool, jobs, title):
    futures = {pool.submit(worker,job):i for i,job in enumerate(jobs)}
    rows = [None]*len(jobs)
    for done,f in enumerate(as_completed(futures),1):
        rows[futures[f]] = f.result()
        print(title,done,'/',len(jobs),flush=True)
    return rows


def summarize(rows):
    base = {tuple(r['case']):r for r in rows if r['policy']=='Myopic'}
    result = {}
    for label in dict.fromkeys(r['policy'] for r in rows):
        selected = [r for r in rows if r['policy']==label]
        result[label] = {}
        for n in (100,150,'all'):
            cell = [r for r in selected if n=='all' or r['case'][1]==n]
            diffs=[]
            for row in cell:
                baseline = base[tuple(row['case'])]
                assert (row['input_hash'],row['i0_hash'])==(baseline['input_hash'],baseline['i0_hash'])
                diffs.append(row['net']-baseline['net'])
            result[label][str(n)] = dict(n=len(cell),net=mean(r['net'] for r in cell),
                gain=mean(diffs),wins=sum(d>1e-7 for d in diffs),losses=sum(d< -1e-7 for d in diffs),
                layout_gains={b:mean(d for d,r in zip(diffs,cell) if r['layout']==b) for b in sorted({r['layout'] for r in cell})})
    return result


def main():
    here = Path(__file__).parent
    sources = [here/f for f in ('run_high_demand_screen.py','high_demand_features.py',
        'high_demand_screen.py','nn_refinement.py','recovery_opportunity.py','run_global_demand_pilot.py','run_multiscale.py')]
    sources += list((here/'e1_sim').glob('*.py'))
    protocol = dict(splits=SPLITS,groups=GROUPS,names=NAMES,cities=2,share=.4,totals=[100,150],
        fixed_review_min=5,source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        shortlist='all subsets including intercept; value-validation MAE, then best remaining R2',
        deployment_choice='compare two shortlisted policies to scalar and Myopic on separate policy-validation layouts; choose simplest within 1 CNY of best. Freeze before confirmation.',
        policy_priority=['Myopic','scalar','shortlist_1','shortlist_2'],formal_test=False)
    path=OUT/'protocol.json'
    if path.exists() and digest(read(path))!=digest(protocol):
        raise RuntimeError('Source or protocol changed. Use a fresh campaign directory.')
    write(path,protocol)
    with ProcessPoolExecutor(max_workers=6) as pool:
        # One world per case is generated once, before concurrent policy reads.
        bank = batch(pool,[(sp,'teacher',None,True) for stage in ('fit','value_validation')
                           for sp in cases(stage)],'bank')
        ranked = screen([r for r in bank if r['layout'] in SPLITS['fit']],
                        [r for r in bank if r['layout'] in SPLITS['value_validation']],GROUPS)
        write(OUT/'all_feature_groups.json',ranked)
        models={'Myopic':None,'scalar':None,**{f'shortlist_{i}':m for i,m in enumerate(ranked['finalists'],1)}}
        for stage in ('policy_validation','confirmation'):
            specs=cases(stage)
            # Baseline first populates new worlds and provides independent ledger checks.
            rows=batch(pool,[(sp,'Myopic',None,False) for sp in specs],stage+' baseline')
            labels=list(models)[1:] if stage=='policy_validation' else list(dict.fromkeys(['scalar',chosen,*[k for k in models if k.startswith('shortlist')]]))
            rows+=batch(pool,[(sp,k,models[k],False) for k in labels if k!='Myopic' for sp in specs],stage+' policies')
            metrics=summarize(rows)
            write(OUT/(stage+'.json'),metrics)
            if stage=='policy_validation':
                top=max(m['all']['net'] for m in metrics.values())
                priority=sorted(models,key=lambda k:(0 if k=='Myopic' else 1 if k=='scalar' else 2,len((models[k] or {}).get('columns',[]))))
                chosen=next(k for k in priority if metrics[k]['all']['net']>=top-1)
                write(OUT/'freeze.json',dict(chosen=chosen,model=models[chosen],shortlisted=models,validation=metrics))
        print('COMPLETE',chosen,metrics[chosen],flush=True)


if __name__=='__main__':
    main()
