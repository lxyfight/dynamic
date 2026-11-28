"""Development retraining on the planned workload; count policy rollouts as episodes.

Five grid evaluations of one exogenous realization are five complete episodes.
Validation/confirmation episodes and scenario counts are recorded separately.
This diagnostic campaign does not freeze the manuscript's 600-instance test.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from run_multiscale import ROOT,CFG,read,write
from e1_sim.initializer import digest
from profit_diagnosis import instance,specification
from recovery_opportunity import run

OUT=ROOT/'tmp/workload_training_20261124'
SEEDS=[17,29,43];LEVELS=[0,24,28,30,36]

def specs(split,s=0):
    rows=[]
    for l in range(2 if split=='train' else 1 if split=='validation' else 3):
        for c in ((2,4,6) if split=='confirmation' else (2,4)):
            for D in (10,20):
                for rho in (.2,.4):
                    sp=specification(f'workload-{split}-{s}',c,l,0,D,D*rho/(1-rho))
                    rows.append(sp)
    return rows

def job(task):
    sp,w=task;path=OUT/'runs'/(digest(task)+'.json')
    if path.exists():return read(path)
    ins=instance(CFG,**sp);row=run(ins,w)
    row.update(spec=sp,theta=w,i0_net=ins['i0_net'],delivery_revenue=ins['delivery_revenue'],
               realized_pickups=ins['realized_pickups'],input_hash=ins['input_hash'])
    write(path,row);return row

def batch(pool,tasks):
    groups={}
    for i,t in enumerate(tasks):groups.setdefault(digest(t),[t,[]])[1].append(i)
    rows=[None]*len(tasks);pending={pool.submit(job,t):ii for t,ii in groups.values()}
    for f in as_completed(pending):
        r=f.result()
        for i in pending[f]:rows[i]=r
    return rows

def selected(means):
    best=max(means.values())
    return min(w for w,v in means.items() if v>=best-1.)

def main():
    protocol=dict(cfg=CFG,levels=LEVELS,seeds=SEEDS,train={s:specs('train',s) for s in SEEDS},
        validation=specs('validation'),confirmation=specs('confirmation'),
        selection='lowest theta within 1 CNY of highest training mean; all seeds retained',
        checkpoints_scenarios=[4,8,12,16],checkpoints_training_episodes=[20,40,60,80],
        episode='one complete simulated operating horizon including accepted obligations and return; five coefficients on one scenario count as five training episodes',
        validation_used_for_selection=False,formal_test=False,
        source={p.name:digest(p.read_text(encoding='utf8')) for p in [Path(__file__),Path(__file__).with_name('profit_diagnosis.py'),Path(__file__).with_name('recovery_opportunity.py')]})
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==read_json_roundtrip(protocol)
    write(path,protocol)
    with ProcessPoolExecutor(max_workers=6) as pool:
        validation=batch(pool,[(sp,w) for sp in specs('validation') for w in LEVELS])
        write(OUT/'validation_all.json',validation)
        training=batch(pool,[(sp,w) for s in SEEDS for sp in specs('train',s) for w in LEVELS])
        write(OUT/'training.json',training)
        curves={};chosen={}
        for s in SEEDS:
            bank=[r for r in training if r['spec'] in specs('train',s)]
            curves[s]=[]
            for n in [4,8,12,16]:
                included=specs('train',s)[:n]
                means={w:float(np.mean([r['net'] for r in bank if r['theta']==w and r['spec'] in included])) for w in LEVELS}
                w=selected(means);vr=[r for r in validation if r['theta']==w];base=[r for r in validation if r['theta']==0]
                curves[s].append(dict(episode=n*len(LEVELS),exogenous_scenarios=n,theta=w,training_means=means,
                    validation_mean=float(np.mean([r['net'] for r in vr])),
                    validation_gain=float(np.mean([a['net']-b['net'] for a,b in zip(vr,base)])),
                    validation_plan_hashes=[r['plans'] for r in vr],optimizer_steps=0))
            chosen[s]=curves[s][-1]['theta']
        write(OUT/'curves.json',curves)
        write(OUT/'freeze.json',dict(chosen=chosen,training_episodes_per_seed=80,
            evaluation_episodes_shared=40,formal_test_frozen=False))
        print('development models selected',chosen,flush=True)
        confirm=batch(pool,[(sp,w) for sp in specs('confirmation') for w in sorted({0,*chosen.values()})])
        write(OUT/'confirmation.json',confirm)
        print('complete',len(training),'training episodes;',len(confirm),'confirmation episodes',flush=True)

def read_json_roundtrip(value):
    import json
    return json.loads(json.dumps(value))

if __name__=='__main__':main()
