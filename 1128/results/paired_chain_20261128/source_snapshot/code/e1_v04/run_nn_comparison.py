"""Prespecified development comparison; never writes manuscript results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,time
import numpy as np
from run_multiscale import CFG,ROOT,write,read
from e1_sim.unknown_demand import make_instance,seed
from e1_sim.initializer import digest
from nn_comparison import execute,fit,NAMES

OUT=ROOT/'tmp/nn_comparison_sortie25_v1'
SEEDS=[17,29,43];LEVELS=[0,24,36,48,60]

def specs(split,s=0,count=48):
    rows=[]
    if split=='confirm':
        cells=[(c,l,d,mu) for c in (2,4,6) for l in range(6) for d in range(2) for mu in (2,4)]
    elif split=='valid':cells=[(c,l,0,mu) for c in (2,4) for l in range(6) for mu in (2,4)]
    else:cells=[((2,4)[(i//2)%2],(i//4)%6,i//24,(2,4)[i%2]) for i in range(count)]
    for c,l,d,mu in cells:
        family=('nn_sortie25_v1',split,s,c)
        rows.append(dict(identity=f'{split}_s{s}_c{c}_l{l}_d{d}_m{mu}',cities=c,deliveries_per_city=4,
            expected_pickups_per_city=mu,layout_seed=seed(*family,'layout',l),day_seed=seed(*family,'day',l,d),pickup_seed=seed(*family,'pickup',l,d,mu)))
    return rows

def job(task):
    sp,model,rng,epsilon=task
    key=digest(dict(spec=sp,model=model,rng=rng,epsilon=epsilon))
    path=OUT/'runs'/(key+'.json')
    if path.exists():return read(path)
    x=make_instance(CFG,**sp);row=execute(x,model,rng,epsilon)
    row['spec']=sp;row['model_kind']=model['kind'];write(path,row)
    return row

def batch(pool,tasks):
    result=[None]*len(tasks);pending={pool.submit(job,t):i for i,t in enumerate(tasks)}
    for f in as_completed(pending):result[pending[f]]=f.result()
    return result

def tasks(sp,model,s=0,epsilon=0):return [(x,model,seed('explore',s,x['identity']),epsilon) for x in sp]

def ci(rows,gains):
    rng=np.random.default_rng(7781);draws=[]
    for c in sorted({r['spec']['cities'] for r in rows}):
        blocks={}
        for row,g in zip(rows,gains):
            if row['spec']['cities']==c:blocks.setdefault(row['spec']['layout_seed'],[]).append(g)
        values=np.array([np.mean(v) for v in blocks.values()]);draws.append(values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1))
    return np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist()

def main():
    start=time.perf_counter()
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('nn_comparison.py'),*Path(__file__).with_name('e1_sim').glob('*.py'),Path(__file__).with_name('recovery_opportunity.py')]}
    protocol=dict(cfg=CFG,source=source,features=NAMES,seeds=SEEDS,train_days_per_seed=48,checkpoints=[12,24,36,48],
        architecture=[len(NAMES),64,64,1],label='G_day - J; J includes accrued ledger; executed actions only; closed rows excluded',
        fit='equal total sample weight per day; train-only mean/std; output scaled by 100 CNY; weighted ridge alpha .01 or 250 full-batch Adam steps lr .002 weight penalty .0001',
        exploration=[.15,.1,.05,.05],selection='per-seed max mean validation net; earliest checkpoint on ties; never choose best seed',
        stability='last 3 checkpoints per-seed validation gain range <=1 CNY/day and adjacent plan agreement >=.9; not required to run diagnostic confirmation; report failure explicitly',
        validation=specs('valid'),confirmation=specs('confirm'),training={str(s):specs('train',s) for s in SEEDS},formal_test=False)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol,'protocol/source changed; use a new run directory'
    write(OUT/'protocol.json',protocol)
    models={};banks={};curves={};valid=specs('valid');zero=dict(kind='zero')
    with ProcessPoolExecutor(max_workers=8) as pool:
        baseline=batch(pool,tasks(valid,zero));print('validation Myopic done',flush=True)
        warmtasks=[t for s in SEEDS for t in tasks(specs('train',s)[:12],zero,s,.15)]
        warm=batch(pool,warmtasks)
        for i,s in enumerate(SEEDS):
            for kind in ('linear','mlp'):banks[(s,kind)]=list(warm[i*12:(i+1)*12]);curves[(s,kind)]=[]
        for iteration in range(4):
            if iteration:
                todo=[]
                for s in SEEDS:
                    for kind in ('linear','mlp'):
                        todo+=tasks(specs('train',s)[12*iteration:12*(iteration+1)],models[(s,kind)],s,[.15,.1,.05,.05][iteration])
                sampled=batch(pool,todo);offset=0
                for s in SEEDS:
                    for kind in ('linear','mlp'):
                        banks[(s,kind)]+=sampled[offset:offset+12];offset+=12
            for s in SEEDS:
                for kind in ('linear','mlp'):models[(s,kind)]=fit(banks[(s,kind)],kind,s)
            evaluated=batch(pool,[t for s in SEEDS for kind in ('linear','mlp') for t in tasks(valid,models[(s,kind)])]);offset=0
            for s in SEEDS:
                for kind in ('linear','mlp'):
                    rows=evaluated[offset:offset+len(valid)];offset+=len(valid)
                    gain=float(np.mean([r['net']-b['net'] for r,b in zip(rows,baseline)]))
                    point=dict(days=12*(iteration+1),gain=gain,plans=[r['plans'] for r in rows],model=models[(s,kind)],training_rmse_cny=models[(s,kind)]['training_rmse_cny'])
                    curves[(s,kind)].append(point)
            write(OUT/'curves.json',{f'{s}_{k}':v for (s,k),v in curves.items()})
            print('checkpoint',12*(iteration+1),{f'{s}_{k}':round(v[-1]['gain'],3) for (s,k),v in curves.items()},flush=True)
        # Scalar policy-search baseline has the same 48 exogenous training days,
        # but five complete policy evaluations per day; report this extra budget.
        scalar=batch(pool,[t for s in SEEDS for w in LEVELS for t in tasks(specs('train',s),dict(kind='scalar',theta=w))])
        chosen={};scalar_scores={};offset=0
        for s in SEEDS:
            means={}
            for w in LEVELS:means[w]=float(np.mean([r['net'] for r in scalar[offset:offset+48]]));offset+=48
            best=max(means.values());w=min(w for w in LEVELS if means[w]>=best-1)
            chosen[f'{s}_scalar']=dict(kind='scalar',theta=w);scalar_scores[s]=means
            for kind in ('linear','mlp'):
                bestpoint=max(curves[(s,kind)],key=lambda p:(p['gain'],-p['days']))
                chosen[f'{s}_{kind}']=bestpoint['model']
        freeze=dict(models=chosen,scalar_training_means=scalar_scores,selection_hash=digest(chosen))
        write(OUT/'freeze.json',freeze);print('frozen',[(k,v.get('theta',v.get('training_days'))) for k,v in chosen.items()],flush=True)
        confirm=specs('confirm');allmodels={'Myopic':zero,**chosen}
        results=batch(pool,[t for model in allmodels.values() for t in tasks(confirm,model)])
    by={name:results[i*len(confirm):(i+1)*len(confirm)] for i,name in enumerate(allmodels)}
    base=by['Myopic'];summary={}
    for kind in ('scalar','linear','mlp'):
        seeds=np.array([[r['net']-b['net'] for r,b in zip(by[f'{s}_{kind}'],base)] for s in SEEDS])
        g=seeds.mean(axis=0);bounds=ci(base,g)
        records=[r for s in SEEDS for r in by[f'{s}_{kind}']]
        decisions=[a for r in records for a in r['rows']];times=[t for r in records for t in r['timings']]
        stability=[]
        if kind!='scalar':
            for s in SEEDS:
                points=curves[(s,kind)][-3:]
                agreement=[float(np.mean([a==b for a,b in zip(p['plans'],q['plans'])])) for p,q in zip(points,points[1:])]
                stability.append(dict(seed=s,range=float(np.ptp([p['gain'] for p in points])),agreement=agreement,
                                      passed=bool(np.ptp([p['gain'] for p in points])<=1 and min(agreement)>=.9)))
        summary[kind]=dict(mean_gain=float(g.mean()),ci95=bounds,seed_gains=seeds.mean(axis=1).tolist(),
            mean_net=float(np.mean([r['net'] for r in records])),win=int(sum(g>1e-7)),loss=int(sum(g< -1e-7)),tie=int(sum(abs(g)<=1e-7)),
            p95_decision_seconds=float(np.quantile(times,.95)),changed_action_fraction=float(np.mean([r['differs'] for r in decisions])),
            same_input_pool_fraction=float(np.mean([r['unique_inputs']==1 for r in decisions])),stability=stability,
            benefit=bool(g.mean()>=1 and bounds[0]>0 and sum(seeds.mean(axis=1)>0)>=2))
    nn_linear=np.mean([[a['net']-b['net'] for a,b in zip(by[f'{s}_mlp'],by[f'{s}_linear'])] for s in SEEDS],axis=0)
    result=dict(policies=summary,myopic_mean=float(np.mean([r['net'] for r in base])),nn_minus_linear=dict(mean=float(nn_linear.mean()),ci95=ci(base,nn_linear)),
        elapsed_seconds=time.perf_counter()-start,confirmation_configs=len(confirm),formal_test=False,
        runs=len(list((OUT/'runs').glob('*.json'))),timing_caveat='Concurrent worker timings; not isolated formal timing benchmark.')
    write(OUT/'confirmation.json',by);write(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
