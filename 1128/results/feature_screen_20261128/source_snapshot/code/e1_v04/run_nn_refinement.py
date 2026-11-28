"""Second development protocol: reject-all control variate and guarded residual learning."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from random import Random
import json,time,hashlib
import numpy as np
from run_multiscale import ROOT,CFG,read,write
from run_nn_comparison import specs as old_specs,ci
from e1_sim.unknown_demand import make_instance,seed,sample_orders,public_law
from e1_sim.multiscale import network
from e1_sim.initializer import digest
from nn_refinement import execute,fit,NAMES

OUT=ROOT/'tmp/nn_comparison_sortie25_v2';OLD=ROOT/'tmp/nn_comparison_sortie25_v1'
SEEDS=[17,29,43];LEVELS=[0,24,26,28,30,32,34,36,48,60];ALPHAS=[0,.1,.25,.5]

def specs(split,s=0):
    if split in ('train','valid'):return old_specs(split,s)
    rows=old_specs('confirm' if split=='confirm' else 'train',s)
    if split=='refine':rows=rows[:12]
    for r in rows:
        r['identity']='refinement_'+r['identity']
        for key in ('layout_seed','day_seed','pickup_seed'):r[key]=seed('nn_v2',split,r[key])
    return rows

def centered(row):
    row=deepcopy(row);sp=row['spec'];law=public_law(sp['cities'],sp['expected_pickups_per_city'])
    orders=list(sample_orders(law,0,Random(sp['pickup_seed']),network(sp['cities'],sp['layout_seed']),CFG).values())
    for r in row['rows']:
        r['raw_y']=row['net']-r['j']
        r['future_rejection_baseline']=sum(o['weight']*CFG['rejection_per_kg'] for o in orders if o['release']>r['t'])
        r['y']=r['raw_y']+r['future_rejection_baseline']
    return row

def job(task):
    sp,model,rng,epsilon=task;key=digest(dict(spec=sp,model=model,rng=rng,epsilon=epsilon));path=OUT/'runs'/(key+'.json')
    if path.exists():return read(path)
    historical=OLD/'runs'/(key+'.json')
    if model['kind'] in ('scalar','zero') and historical.exists():row=centered(read(historical))
    else:
        row=execute(make_instance(CFG,**sp),model,rng,epsilon);row['spec']=sp;row['model_kind']=model['kind']
    write(path,row);return row

def tasks(rows,model,s=0,epsilon=0):return [(r,model,seed('explore',s,r['identity']),epsilon) for r in rows]
def batch(pool,ts):
    rows=[None]*len(ts);groups={};unique={}
    for i,t in enumerate(ts):
        key=digest(t);groups.setdefault(key,[]).append(i);unique[key]=t
    pending={pool.submit(job,t):key for key,t in unique.items()}
    for f in as_completed(pending):
        row=f.result()
        for i in groups[pending[f]]:rows[i]=row
    return rows

def correction(bank,kind,s,theta):
    days=deepcopy(bank)
    for d in days:
        for r in d['rows']:r['y']-=theta*r['x'][0]
    return fit(days,kind,s)

def main():
    begin=time.perf_counter();zero=dict(kind='zero');valid=specs('valid')
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('nn_refinement.py')]}
    protocol=dict(cfg=CFG,source=sources,parent_protocol_hash=digest(read(OLD/'protocol.json')),features=NAMES,levels=LEVELS,seeds=SEEDS,alphas=ALPHAS,
        target='Y = G-J+realized future rejection loss. This loss is used only offline as a control variate; its expectation depends only on known time/public law and is common to all candidates. Scores omit that common baseline.',
        model='theta*coverage + alpha*learned correction; Linear and MLP share inputs and labels; alpha=0 is explicit scalar fallback, never evidence of NN benefit.',
        training='48 teacher days + 12 new on-policy days per seed/model; equal day weighting; same fixed optimizer settings as first protocol',
        selection='scalar coefficient by training mean, smallest within 1 CNY; correction alpha/checkpoint by validation mean, smallest alpha within 1 CNY of best, earliest stage on ties',
        confirm=specs('confirm'),valid=valid,formal_test=False)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    write(OUT/'protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=8) as pool:
        baseline=batch(pool,tasks(valid,zero))
        trained=batch(pool,[t for s in SEEDS for w in LEVELS for t in tasks(specs('train',s),dict(kind='scalar',theta=w))])
        teachers={};banks={};means={};offset=0
        for s in SEEDS:
            bank={}
            for w in LEVELS:bank[w]=trained[offset:offset+48];offset+=48
            means[s]={w:float(np.mean([r['net'] for r in rows])) for w,rows in bank.items()}
            top=max(means[s].values());w=min(w for w in LEVELS if means[s][w]>=top-1)
            teachers[s]=dict(kind='scalar',theta=w)
            for kind in ('linear','mlp'):banks[(s,kind)]=bank[w]
        write(OUT/'teachers.json',dict(models=teachers,means=means));print('teachers',teachers,flush=True)
        histories={};current={}
        for stage in (0,1):
            if stage:
                sampled=batch(pool,[t for s in SEEDS for k in ('linear','mlp') for t in tasks(specs('refine',s),current[(s,k)],s,.05)])
                off=0
                for s in SEEDS:
                    for k in ('linear','mlp'):banks[(s,k)]=banks[(s,k)]+sampled[off:off+12];off+=12
            models={}
            for s in SEEDS:
                for k in ('linear','mlp'):
                    w=teachers[s]['theta'];fitted=correction(banks[(s,k)],k,s,w)
                    for a in ALPHAS:models[(s,k,a)]=dict(kind='residual',theta=w,alpha=a,correction=fitted) if a else teachers[s]
            evaluated=batch(pool,[t for model in models.values() for t in tasks(valid,model)])
            off=0
            for (s,k,a),model in models.items():
                rows=evaluated[off:off+len(valid)];off+=len(valid)
                gain=float(np.mean([r['net']-b['net'] for r,b in zip(rows,baseline)]))
                histories.setdefault((s,k),[]).append(dict(stage=stage,alpha=a,gain=gain,model=model))
            for key,history in histories.items():
                top=max(p['gain'] for p in history)
                chosen=min((p for p in history if p['gain']>=top-1),key=lambda p:(p['alpha'],p['stage']))
                current[key]=chosen['model']
            write(OUT/'validation.json',{f'{s}_{k}':v for (s,k),v in histories.items()})
            print('stage',stage,{f'{s}_{k}':(current[(s,k)].get('alpha',0),round(max(p['gain'] for p in v),3)) for (s,k),v in histories.items()},flush=True)
        chosen={'Myopic':zero}
        for s in SEEDS:
            chosen[f'{s}_scalar']=teachers[s]
            for k in ('linear','mlp'):chosen[f'{s}_{k}']=current[(s,k)]
        write(OUT/'freeze.json',dict(models=chosen,hash=digest(chosen)));print('models frozen before fresh confirmation',flush=True)
        confirm=specs('confirm');rs=batch(pool,[t for model in chosen.values() for t in tasks(confirm,model)])
    by={k:rs[i*len(confirm):(i+1)*len(confirm)] for i,k in enumerate(chosen)};base=by['Myopic'];result={}
    for k in ('scalar','linear','mlp'):
        gains=np.array([[a['net']-b['net'] for a,b in zip(by[f'{s}_{k}'],base)] for s in SEEDS]);g=gains.mean(axis=0)
        rows=[r for s in SEEDS for r in by[f'{s}_{k}']];bounds=ci(base,g)
        result[k]=dict(mean_net=float(np.mean([r['net'] for r in rows])),mean_gain=float(g.mean()),ci95=bounds,
             seed_gains=gains.mean(axis=1).tolist(),alpha=[chosen[f'{s}_{k}'].get('alpha',0) for s in SEEDS],
             benefit=bool(g.mean()>=1 and bounds[0]>0 and sum(gains.mean(axis=1)>0)>=2),
             mean_decisions=float(np.mean([len(r['rows']) for r in rows])),
             p95_decision_seconds=float(np.quantile([t for r in rows for t in r['timings']],.95)))
    diff=np.mean([[a['net']-b['net'] for a,b in zip(by[f'{s}_mlp'],by[f'{s}_linear'])] for s in SEEDS],axis=0)
    output=dict(policies=result,myopic_mean=float(np.mean([r['net'] for r in base])),nn_minus_linear=dict(mean=float(diff.mean()),ci95=ci(base,diff)),
        confirmation_configs=len(confirm),elapsed_seconds=time.perf_counter()-begin,formal_test=False,stable_training_proven=False,
        caveat='Guarded residual development comparison, not pure NN superiority; alpha=0 uses scalar baseline; timing under concurrent workers.')
    write(OUT/'confirmation.json',by);write(OUT/'result.json',output);print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()
